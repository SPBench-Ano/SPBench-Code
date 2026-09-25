#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
按炮点比例把一个完整 SEG-Y 拆成两个 SEG-Y 文件。

特点：
1. 同一个炮点下的所有 trace 一定进入同一个输出文件；
2. 默认按 SourceX/SourceY 识别炮点；
3. 流式读写，不把整份地震数据加载到内存；
4. 输出文件默认命名为：
       原文件名_shotper{比例}.sgy
       原文件名_shotper{1-比例}.sgy

示例：
    python3 split_segy_by_shot_ratio.py input.sgy --shotper 0.7
    python3 split_segy_by_shot_ratio.py input.sgy --shotper 0.8 --seed 123
    python3 split_segy_by_shot_ratio.py input.sgy --shot-key shot_line_stake
"""

from __future__ import annotations

import argparse
import random
import struct
import sys
import time
from pathlib import Path
from typing import BinaryIO, Callable, Dict, List, Optional, Sequence, Set, Tuple


TEXTUAL_HEADER_BYTES = 3200
BINARY_HEADER_BYTES = 400
TRACE_HEADER_BYTES = 240

# SEG-Y 常见样点格式码 -> 每个样点占用字节数
SAMPLE_FORMAT_BYTES = {
    1: 4,   # 4-byte IBM float
    2: 4,   # 4-byte int
    3: 2,   # 2-byte int
    4: 4,   # 4-byte fixed-point with gain
    5: 4,   # 4-byte IEEE float
    6: 8,   # 8-byte IEEE float
    7: 3,   # 3-byte int
    8: 1,   # 1-byte int
    9: 8,   # 8-byte int
    10: 4,  # 4-byte unsigned int
    11: 2,  # 2-byte unsigned int
    12: 8,  # 8-byte unsigned int
    15: 3,  # 3-byte unsigned int
    16: 1,  # 1-byte unsigned int
}

# 1-based 道头字节位置，读取 4 字节大端 int32。
SHOT_KEY_BYTE_POS = {
    "source_xy": (73, 77),          # SourceX, SourceY
    "shot_line_stake": (17, 21),    # 炮线, 炮桩号
    "shot_line_no": (17, 25),       # 炮线, 炮号
    "shot_line": (17,),
    "shot_no": (25,),
}    #for segc3
'''SHOT_KEY_BYTE_POS = {
    "source_xy": (73, 77),          # SourceX, SourceY
    "shot_line_stake": (225, 225),    # 炮线, 炮桩号
    "shot_line_no": (221, 25),       # 炮线, 炮号
    "shot_line": (221,),
    "shot_no": (25,),
}''' #for sw06_dongfang

ShotKey = Tuple[int, ...]
ShotKeyReader = Callable[[bytes], ShotKey]


def _u16be(buf: bytes, offset: int) -> int:
    return struct.unpack(">H", buf[offset:offset + 2])[0]


def _i32be_from_trace_header(trace_header: bytes, pos1b: int) -> int:
    if pos1b < 1 or pos1b + 3 > TRACE_HEADER_BYTES:
        raise ValueError(f"道头 int32 字节位置越界: {pos1b}")
    offset = pos1b - 1
    return struct.unpack(">i", trace_header[offset:offset + 4])[0]


def _format_percent(numerator: int, denominator: int) -> str:
    if denominator <= 0:
        return "0.00%"
    return f"{(numerator / denominator) * 100.0:.2f}%"


def _format_ratio_tag(value: float) -> str:
    text = f"{value:.12g}"
    if "e" in text or "E" in text:
        text = f"{value:.12f}".rstrip("0").rstrip(".")
    return text


def _parse_shot_bytes(text: Optional[str]) -> Optional[Tuple[int, ...]]:
    if text is None:
        return None
    positions = tuple(int(item.strip()) for item in text.split(",") if item.strip())
    if not positions:
        raise ValueError("--shot-bytes 不能为空")
    for pos in positions:
        if pos < 1 or pos + 3 > TRACE_HEADER_BYTES:
            raise ValueError(f"--shot-bytes 中的 int32 字节位置越界: {pos}")
    return positions


def _build_shot_key_reader(shot_key: str, shot_bytes: Optional[str]) -> ShotKeyReader:
    custom_positions = _parse_shot_bytes(shot_bytes)
    if custom_positions is not None:
        positions = custom_positions
    else:
        if shot_key not in SHOT_KEY_BYTE_POS:
            raise ValueError(
                f"未知 shot-key: {shot_key}，可选: {sorted(SHOT_KEY_BYTE_POS)}"
            )
        positions = SHOT_KEY_BYTE_POS[shot_key]

    def read_key(trace_header: bytes) -> ShotKey:
        return tuple(_i32be_from_trace_header(trace_header, pos) for pos in positions)

    return read_key


def read_file_headers(fin: BinaryIO) -> tuple[bytes, int, int, int]:
    header_block = fin.read(TEXTUAL_HEADER_BYTES + BINARY_HEADER_BYTES)
    if len(header_block) != TEXTUAL_HEADER_BYTES + BINARY_HEADER_BYTES:
        raise ValueError("输入文件不是合法的 SEG-Y：文件头长度不足 3600 字节。")

    binary_header = header_block[TEXTUAL_HEADER_BYTES:]
    ns_from_bin = _u16be(binary_header, 20)
    format_code = _u16be(binary_header, 24)
    ext_text_headers = _u16be(binary_header, 304)

    if ext_text_headers > 0:
        ext_bytes = ext_text_headers * TEXTUAL_HEADER_BYTES
        ext_block = fin.read(ext_bytes)
        if len(ext_block) != ext_bytes:
            raise ValueError(
                "输入文件的扩展文本头长度不足，文件可能损坏，"
                f"期望 {ext_bytes} 字节，实际读到 {len(ext_block)} 字节。"
            )
        header_block += ext_block

    return header_block, ns_from_bin, format_code, ext_text_headers


def read_trace_payload(fin: BinaryIO, trace_header: bytes, ns_from_bin: int, bytes_per_sample: int) -> bytes:
    ns_from_trace = _u16be(trace_header, 114)
    ns = ns_from_trace if ns_from_trace > 0 else ns_from_bin
    if ns <= 0:
        raise ValueError(
            f"无法确定 trace 采样点数，bin ns={ns_from_bin}, trace ns={ns_from_trace}。"
        )

    payload_bytes = int(ns) * bytes_per_sample
    payload = fin.read(payload_bytes)
    if len(payload) != payload_bytes:
        raise ValueError(
            f"trace 数据长度不足，期望 {payload_bytes} 字节，实际读到 {len(payload)} 字节。"
        )
    return payload


def scan_shots(
    input_path: Path,
    shot_key_reader: ShotKeyReader,
    report_every: int = 100000,
) -> tuple[bytes, int, int, int, List[ShotKey], Dict[ShotKey, int], int]:
    start_time = time.time()
    shot_order: List[ShotKey] = []
    shot_counts: Dict[ShotKey, int] = {}
    total_traces = 0

    with input_path.open("rb") as fin:
        header_block, ns_from_bin, format_code, ext_text_headers = read_file_headers(fin)
        if format_code not in SAMPLE_FORMAT_BYTES:
            raise ValueError(f"暂不支持的 SEG-Y 样点格式码: {format_code}")
        bytes_per_sample = SAMPLE_FORMAT_BYTES[format_code]

        while True:
            trace_header = fin.read(TRACE_HEADER_BYTES)
            if not trace_header:
                break
            if len(trace_header) != TRACE_HEADER_BYTES:
                raise ValueError(
                    f"第 {total_traces} 道的道头长度不足 240 字节，文件可能损坏。"
                )

            key = shot_key_reader(trace_header)
            if key not in shot_counts:
                shot_counts[key] = 0
                shot_order.append(key)
            shot_counts[key] += 1

            read_trace_payload(fin, trace_header, ns_from_bin, bytes_per_sample)
            total_traces += 1

            if report_every > 0 and total_traces % report_every == 0:
                elapsed = time.time() - start_time
                print(
                    "[scan] "
                    f"已扫描 {total_traces} 道 | 唯一炮点 {len(shot_order)} | "
                    f"用时 {elapsed:.1f} s",
                    flush=True,
                )

    return (
        header_block,
        ns_from_bin,
        format_code,
        ext_text_headers,
        shot_order,
        shot_counts,
        total_traces,
    )


def choose_shot_sets(
    shot_order: Sequence[ShotKey],
    shotper: float,
    seed: int,
    shuffle: bool,
) -> tuple[Set[ShotKey], Set[ShotKey]]:
    n_shots = len(shot_order)
    if n_shots < 2:
        raise ValueError(f"唯一炮点数为 {n_shots}，无法拆成两个非空文件。")

    n_first = int(round(n_shots * shotper))
    n_first = max(1, min(n_shots - 1, n_first))

    selected = list(shot_order)
    if shuffle:
        rng = random.Random(seed)
        rng.shuffle(selected)

    first = set(selected[:n_first])
    second = set(selected[n_first:])
    return first, second


def build_output_paths(input_path: Path, output_dir: Path, shotper: float) -> tuple[Path, Path]:
    suffix = input_path.suffix or ".sgy"
    stem = input_path.stem
    first_tag = _format_ratio_tag(shotper)
    second_tag = _format_ratio_tag(1.0 - shotper)
    first_path = output_dir / f"{stem}_shotper{first_tag}{suffix}"
    second_path = output_dir / f"{stem}_shotper{second_tag}{suffix}"

    # 例如 shotper=0.5 时两个文件名会相同，这时只在末尾加 part 标识避免覆盖。
    if first_path == second_path:
        first_path = output_dir / f"{stem}_shotper{first_tag}_part1{suffix}"
        second_path = output_dir / f"{stem}_shotper{second_tag}_part2{suffix}"
    return first_path, second_path


def split_traces(
    input_path: Path,
    first_output: Path,
    second_output: Path,
    first_shots: Set[ShotKey],
    second_shots: Set[ShotKey],
    shot_key_reader: ShotKeyReader,
    header_block: bytes,
    ns_from_bin: int,
    format_code: int,
    overwrite: bool = False,
    report_every: int = 100000,
) -> tuple[int, int]:
    if first_output.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {first_output}，如需覆盖请加 --overwrite")
    if second_output.exists() and not overwrite:
        raise FileExistsError(f"输出文件已存在: {second_output}，如需覆盖请加 --overwrite")
    if first_output == second_output:
        raise ValueError(f"两个输出文件路径相同: {first_output}")

    first_output.parent.mkdir(parents=True, exist_ok=True)
    second_output.parent.mkdir(parents=True, exist_ok=True)

    bytes_per_sample = SAMPLE_FORMAT_BYTES[format_code]
    first_traces = 0
    second_traces = 0
    total_traces = 0
    start_time = time.time()

    with input_path.open("rb") as fin, first_output.open("wb") as fout_a, second_output.open("wb") as fout_b:
        read_file_headers(fin)
        fout_a.write(header_block)
        fout_b.write(header_block)

        while True:
            trace_header = fin.read(TRACE_HEADER_BYTES)
            if not trace_header:
                break
            if len(trace_header) != TRACE_HEADER_BYTES:
                raise ValueError(
                    f"第 {total_traces} 道的道头长度不足 240 字节，文件可能损坏。"
                )

            payload = read_trace_payload(fin, trace_header, ns_from_bin, bytes_per_sample)
            key = shot_key_reader(trace_header)

            if key in first_shots:
                fout_a.write(trace_header)
                fout_a.write(payload)
                first_traces += 1
            elif key in second_shots:
                fout_b.write(trace_header)
                fout_b.write(payload)
                second_traces += 1
            else:
                raise RuntimeError(f"第 {total_traces} 道的炮点键未出现在拆分集合中: {key}")

            total_traces += 1
            if report_every > 0 and total_traces % report_every == 0:
                elapsed = time.time() - start_time
                print(
                    "[write] "
                    f"已写入 {total_traces} 道 | "
                    f"第一份 {first_traces} | 第二份 {second_traces} | "
                    f"用时 {elapsed:.1f} s",
                    flush=True,
                )

    return first_traces, second_traces


def split_segy_by_shot_ratio(
    input_segy: Path,
    shotper: float = 0.5,
    output_dir: Optional[Path] = None,
    shot_key: str = "source_xy",
    shot_bytes: Optional[str] = None,
    seed: int = 42,
    shuffle: bool = True,
    overwrite: bool = False,
    report_every: int = 100000,
) -> tuple[Path, Path]:
    if not input_segy.exists():
        raise FileNotFoundError(f"输入 SEG-Y 不存在: {input_segy}")
    if not input_segy.is_file():
        raise ValueError(f"输入路径不是文件: {input_segy}")
    if not 0.0 < shotper < 1.0:
        raise ValueError(f"shotper 必须在 (0, 1) 之间，当前为: {shotper}")

    output_dir = output_dir or input_segy.parent
    output_a, output_b = build_output_paths(input_segy, output_dir, shotper)
    shot_key_reader = _build_shot_key_reader(shot_key, shot_bytes)

    print(
        "[start] "
        f"输入: {input_segy} | shotper={shotper} | "
        f"shot_key={shot_key if shot_bytes is None else shot_bytes}"
    )
    (
        header_block,
        ns_from_bin,
        format_code,
        ext_text_headers,
        shot_order,
        shot_counts,
        total_traces,
    ) = scan_shots(
        input_path=input_segy,
        shot_key_reader=shot_key_reader,
        report_every=report_every,
    )

    first_shots, second_shots = choose_shot_sets(
        shot_order=shot_order,
        shotper=shotper,
        seed=seed,
        shuffle=shuffle,
    )
    first_trace_expected = sum(shot_counts[key] for key in first_shots)
    second_trace_expected = sum(shot_counts[key] for key in second_shots)

    print(
        "[split] "
        f"总道数 {total_traces} | 唯一炮点 {len(shot_order)} | "
        f"样点格式码 {format_code} | bin 头采样点数 {ns_from_bin} | "
        f"扩展文本头 {ext_text_headers}"
    )
    print(
        "[split] "
        f"第一份炮点 {len(first_shots)} ({_format_percent(len(first_shots), len(shot_order))})，"
        f"预计 trace {first_trace_expected}；"
        f"第二份炮点 {len(second_shots)} ({_format_percent(len(second_shots), len(shot_order))})，"
        f"预计 trace {second_trace_expected}"
    )

    first_traces, second_traces = split_traces(
        input_path=input_segy,
        first_output=output_a,
        second_output=output_b,
        first_shots=first_shots,
        second_shots=second_shots,
        shot_key_reader=shot_key_reader,
        header_block=header_block,
        ns_from_bin=ns_from_bin,
        format_code=format_code,
        overwrite=overwrite,
        report_every=report_every,
    )

    if first_traces != first_trace_expected or second_traces != second_trace_expected:
        raise RuntimeError(
            "写入 trace 数与扫描阶段不一致："
            f"first {first_traces}/{first_trace_expected}, "
            f"second {second_traces}/{second_trace_expected}"
        )

    print(f"[done] 第一份: {output_a} | trace={first_traces}")
    print(f"[done] 第二份: {output_b} | trace={second_traces}")
    return output_a, output_b


def build_argparser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="按唯一炮点比例把一个完整 SEG-Y 拆成两个 SEG-Y 文件。"
    )
    parser.add_argument("input_segy", type=Path, help="输入 SEG-Y 文件路径")
    parser.add_argument(
        "--shotper",
        type=float,
        default=0.2,
        help="第一份文件保留的炮点比例，范围 (0,1)，默认 0.5",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="输出目录，默认与输入 SEG-Y 同目录",
    )
    parser.add_argument(
        "--shot-key",
        choices=sorted(SHOT_KEY_BYTE_POS),
        default="source_xy",
        help="用于识别炮点的道头字段组合，默认 source_xy",
    )
    parser.add_argument(
        "--shot-bytes",
        default=None,
        help=(
            "自定义炮点键的 1-based int32 道头字节位置，逗号分隔；"
            "例如 73,77。设置后会覆盖 --shot-key"
        ),
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="随机拆分炮点的种子，默认 42",
    )
    parser.add_argument(
        "--ordered",
        action="store_true",
        help="按文件中炮点首次出现顺序拆分，不随机打乱",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="如果输出文件已存在，则覆盖它",
    )
    parser.add_argument(
        "--report-every",
        type=int,
        default=100000,
        help="每处理多少道打印一次进度，设为 0 表示不打印进度",
    )
    return parser


def main() -> int:
    parser = build_argparser()
    args = parser.parse_args()

    try:
        split_segy_by_shot_ratio(
            input_segy=args.input_segy,
            shotper=args.shotper,
            output_dir=args.output_dir,
            shot_key=args.shot_key,
            shot_bytes=args.shot_bytes,
            seed=args.seed,
            shuffle=not args.ordered,
            overwrite=args.overwrite,
            report_every=args.report_every,
        )
        return 0
    except Exception as exc:
        print(f"[error] {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
