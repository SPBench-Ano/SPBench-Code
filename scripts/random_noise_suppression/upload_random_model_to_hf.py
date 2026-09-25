"""
Upload random-noise-suppression checkpoints (best.pt + config.yaml) to a Hugging Face repository.

Default mapping example:
    /data/pcdisk0/data/results/random_noise/random_noise_unet_base_gaussian_snr0_seed42/checkpoints/best.pt
    -> models/unet/gaussian_snr0_seed42/best.pt

    /data/pcdisk0/data/results/random_noise/random_noise_unet_base_gaussian_snr0_seed42/config.yaml
    -> models/unet/gaussian_snr0_seed42/config.yaml

Usage:
    export HF_NAMESPACE=SPBench
    export HF_TOKEN="your_hf_token"
    python scripts/random_noise_suppression/upload_random_model_to_hf.py --dry-run

Optional:
    --repo-name NAME       HF repo name (default: random-noise-attenuation)
    --results-dir PATH     Override results root
    --models M1 M2 ...     Upload only selected model groups in the given order
    --dry-run              Scan and print what would be uploaded without uploading
    --no-model-card        Skip uploading README.md
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

from huggingface_hub import HfApi, create_repo, hf_hub_download, upload_file

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

SCRIPT_DIR = Path(__file__).resolve().parent
RESULTS_ROOT = "/data/pcdisk0/data/results/random_noise"
DEFAULT_REPO = "random-noise-attenuation"
DEFAULT_MODEL_CARD = SCRIPT_DIR / "README_models.md"

FOLDER_PATTERN = re.compile(
    r"^random_noise_(?P<model>.+?)_base_(?P<noise_kind>[A-Za-z0-9_-]+)"
    r"_snr(?P<snr_tag>neg\d+(?:\.\d+)?|\d+(?:\.\d+)?)_seed(?P<seed>\d+)$"
)
STATS_PATTERN = re.compile(
    r"^random_noise_(?P<model>.+?)_base_(?P<noise_kind>[A-Za-z0-9_-]+)"
    r"_snr(?P<snr_tag>neg\d+(?:\.\d+)?|\d+(?:\.\d+)?)_seed_stats$"
)
AUTO_RESULTS_BEGIN = "<!-- AUTO_RESULTS_BEGIN -->"
AUTO_RESULTS_END = "<!-- AUTO_RESULTS_END -->"
AUTO_RESULTS_DATA_BEGIN = "<!-- AUTO_RESULTS_DATA_BEGIN"
AUTO_RESULTS_DATA_END = "AUTO_RESULTS_DATA_END -->"


def build_order_map(models: list[str] | None) -> dict[str, int]:
    if not models:
        return {}
    return {model: i for i, model in enumerate(models)}


def model_rank(model: str, order_map: dict[str, int] | None = None) -> tuple[int, str]:
    if order_map and model in order_map:
        return (0, str(order_map[model]))
    return (1, model)


def build_display_map(display_items: list[str] | None) -> dict[str, str]:
    mapping: dict[str, str] = {}
    for item in display_items or []:
        if "=" not in item:
            raise ValueError(f"Invalid --display-name item: {item!r}. Expected format model=Display Name")
        model, display = item.split("=", 1)
        model = model.strip()
        display = display.strip()
        if not model or not display:
            raise ValueError(f"Invalid --display-name item: {item!r}. Expected format model=Display Name")
        mapping[model] = display
    return mapping


def model_display(model: str, display_map: dict[str, str] | None = None) -> str:
    if display_map and model in display_map:
        return display_map[model]
    return model


def scan_results(
    results_dir: str,
    allowed_models: list[str] | None = None,
    order_map: dict[str, int] | None = None,
) -> list[dict]:
    """Scan results directory and return uploadable experiment entries."""
    entries: list[dict] = []
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return entries

    allowed = set(allowed_models) if allowed_models else None

    for child in results_path.iterdir():
        if not child.is_dir():
            continue
        m = FOLDER_PATTERN.match(child.name)
        if not m:
            continue

        model = m.group("model")
        if allowed is not None and model not in allowed:
            continue

        best_pt = child / "checkpoints" / "best.pt"
        config_yaml = child / "config.yaml"
        entries.append(
            {
                "folder": child.name,
                "path": str(child),
                "model": model,
                "noise_kind": m.group("noise_kind"),
                "snr_tag": m.group("snr_tag"),
                "seed": m.group("seed"),
                "best_pt": str(best_pt) if best_pt.is_file() else None,
                "config_yaml": str(config_yaml) if config_yaml.is_file() else None,
            }
        )

    entries.sort(
        key=lambda x: (
            model_rank(x["model"], order_map),
            x["noise_kind"],
            x["snr_tag"],
            int(x["seed"]),
        )
    )
    return entries


def hf_path(entry: dict, filename: str) -> str:
    """Construct the path inside the HF repo."""
    subdir = f"{entry['model']}/{entry['noise_kind']}_snr{entry['snr_tag']}_seed{entry['seed']}"
    return f"models/{subdir}/{filename}"


def summarize_entries(entries: list[dict], order_map: dict[str, int] | None, display_map: dict[str, str] | None) -> list[str]:
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["model"]] = counts.get(e["model"], 0) + 1
    lines = []
    for model in sorted(counts.keys(), key=lambda m: model_rank(m, order_map)):
        lines.append(f"  - {model_display(model, display_map)}: {counts[model]} experiment folder(s)")
    return lines


def _snr_sort_key(snr_tag: str) -> tuple[int, float]:
    if snr_tag.startswith("neg"):
        return (0, -float(snr_tag[3:]))
    return (1, float(snr_tag))


def _fmt_mean_std(item: dict[str, float], digits: int = 4) -> str:
    return f"{item['mean']:.{digits}f}±{item['std']:.{digits}f}"


def scan_aggregate_stats(
    results_dir: str,
    allowed_models: list[str] | None = None,
    order_map: dict[str, int] | None = None,
) -> list[dict]:
    """Scan *_seed_stats directories and load aggregate mean/std JSON files."""
    stats_entries: list[dict] = []
    results_path = Path(results_dir)
    if not results_path.is_dir():
        return stats_entries

    allowed = set(allowed_models) if allowed_models else None

    for child in results_path.iterdir():
        if not child.is_dir():
            continue
        m = STATS_PATTERN.match(child.name)
        if not m:
            continue
        model = m.group("model")
        if allowed is not None and model not in allowed:
            continue

        summary_path = child / "metrics_summary_mean_std.json"
        if not summary_path.is_file():
            continue
        try:
            payload = json.loads(summary_path.read_text(encoding="utf-8"))
        except Exception as exc:
            logger.warning("Skip unreadable stats file %s: %s", summary_path, exc)
            continue

        stats_entries.append(
            {
                "model": model,
                "noise_kind": m.group("noise_kind"),
                "snr_tag": m.group("snr_tag"),
                "summary_path": str(summary_path),
                "payload": payload,
            }
        )

    stats_entries.sort(
        key=lambda x: (
            x["noise_kind"],
            _snr_sort_key(x["snr_tag"]),
            model_rank(x["model"], order_map),
        )
    )
    return stats_entries


def build_results_payload(stats_entries: list[dict]) -> dict:
    """Convert aggregate stats JSON files into a structured payload for README generation."""
    payload: dict = {"groups": {}}
    for entry in stats_entries:
        noise_kind = entry["noise_kind"]
        snr_tag = entry["snr_tag"]
        group = payload["groups"].setdefault(noise_kind, {})
        slot = group.setdefault(snr_tag, {"raw": None, "models": {}})

        raw = entry["payload"].get("noisy")
        den = entry["payload"].get("denoised")
        if raw is not None and slot["raw"] is None:
            slot["raw"] = raw
        if den is not None:
            slot["models"][entry["model"]] = den
    return payload


def merge_results_payload(existing: dict | None, current: dict) -> dict:
    """Merge existing README payload with current local results payload."""
    if not existing:
        return current

    merged = {"groups": {}}
    noise_kinds = set(existing.get("groups", {}).keys()) | set(current.get("groups", {}).keys())
    for noise_kind in noise_kinds:
        merged["groups"][noise_kind] = {}
        old_group = existing.get("groups", {}).get(noise_kind, {})
        new_group = current.get("groups", {}).get(noise_kind, {})
        snr_tags = set(old_group.keys()) | set(new_group.keys())
        for snr_tag in snr_tags:
            old_slot = old_group.get(snr_tag, {"raw": None, "models": {}})
            new_slot = new_group.get(snr_tag, {"raw": None, "models": {}})
            merged["groups"][noise_kind][snr_tag] = {
                "raw": new_slot.get("raw") or old_slot.get("raw"),
                "models": dict(old_slot.get("models", {})),
            }
            merged["groups"][noise_kind][snr_tag]["models"].update(new_slot.get("models", {}))
    return merged


def extract_existing_results_payload(repo_id: str, token: str | None) -> dict | None:
    """Read current remote README.md and extract embedded auto-results payload if present."""
    try:
        readme_path = hf_hub_download(repo_id=repo_id, filename="README.md", token=token)
    except Exception:
        return None

    try:
        text = Path(readme_path).read_text(encoding="utf-8")
    except Exception:
        return None

    start = text.find(AUTO_RESULTS_DATA_BEGIN)
    end = text.find(AUTO_RESULTS_DATA_END)
    if start == -1 or end == -1 or end <= start:
        return None

    blob = text[start + len(AUTO_RESULTS_DATA_BEGIN):end].strip()
    try:
        return json.loads(blob)
    except Exception:
        return None


def generate_results_tables(payload: dict, order_map: dict[str, int] | None, display_map: dict[str, str] | None) -> str:
    """Generate markdown result tables from structured payload."""
    groups = payload.get("groups", {})
    if not groups:
        return "## Results\n\nNo aggregate evaluation results were found yet.\n"

    sections = ["## Results", ""]
    sections.append(
        "Mean ± std over available seeds, computed from `*_seed_stats/metrics_summary_mean_std.json`."
    )
    sections.append(
        "Metrics are reported in the normalized domain. `Raw (noisy)` is the synthetic noisy input before denoising."
    )
    sections.append("")

    for noise_kind in sorted(groups.keys()):
        sections.append(f"### {noise_kind.capitalize()} Noise")
        sections.append("")
        for snr_tag in sorted(groups[noise_kind].keys(), key=_snr_sort_key):
            slot = groups[noise_kind][snr_tag]
            snr_label = f"-{snr_tag[3:]}" if snr_tag.startswith("neg") else snr_tag
            sections.append(f"#### SNR {snr_label} dB")
            sections.append("")
            sections.append("| Method | SNR (dB) | PSNR (dB) | SSIM | MAE | MSE | RMSE |")
            sections.append("|---|:---:|:---:|:---:|:---:|:---:|:---:|")

            raw = slot.get("raw")
            if raw is not None:
                sections.append(
                    "| Raw (noisy) | "
                    f"{_fmt_mean_std(raw['snr'])} | "
                    f"{_fmt_mean_std(raw['psnr'])} | "
                    f"{_fmt_mean_std(raw['ssim'])} | "
                    f"{_fmt_mean_std(raw['mae'], 6)} | "
                    f"{_fmt_mean_std(raw['mse'], 6)} | "
                    f"{_fmt_mean_std(raw['rmse'], 6)} |"
                )

            for model in sorted(slot.get("models", {}).keys(), key=lambda m: model_rank(m, order_map)):
                den = slot["models"][model]
                sections.append(
                    f"| {model_display(model, display_map)} | "
                    f"{_fmt_mean_std(den['snr'])} | "
                    f"{_fmt_mean_std(den['psnr'])} | "
                    f"{_fmt_mean_std(den['ssim'])} | "
                    f"{_fmt_mean_std(den['mae'], 6)} | "
                    f"{_fmt_mean_std(den['mse'], 6)} | "
                    f"{_fmt_mean_std(den['rmse'], 6)} |"
                )
            sections.append("")

    return "\n".join(sections).rstrip() + "\n"


def build_model_card(
    model_card_path: Path,
    stats_entries: list[dict],
    repo_id: str,
    token: str | None,
    order_map: dict[str, int] | None,
    display_map: dict[str, str] | None,
) -> bytes | None:
    if not model_card_path.is_file():
        logger.warning("Model card template not found, skip upload: %s", model_card_path)
        return None
    template = model_card_path.read_text(encoding="utf-8")
    current_payload = build_results_payload(stats_entries)
    existing_payload = extract_existing_results_payload(repo_id, token)
    merged_payload = merge_results_payload(existing_payload, current_payload)
    results_md = generate_results_tables(merged_payload, order_map, display_map)
    payload_blob = json.dumps(merged_payload, ensure_ascii=False, indent=2)
    auto_block = (
        f"{AUTO_RESULTS_BEGIN}\n"
        f"{results_md.rstrip()}\n\n"
        f"{AUTO_RESULTS_DATA_BEGIN}\n{payload_blob}\n{AUTO_RESULTS_DATA_END}\n"
        f"{AUTO_RESULTS_END}"
    )
    final_md = template.rstrip() + "\n\n" + auto_block
    return final_md.encode("utf-8")


def upload_model_card(api: HfApi, repo_id: str, token: str, content: bytes | None) -> bool:
    if content is None:
        return False
    try:
        api.upload_file(
            path_or_fileobj=content,
            path_in_repo="README.md",
            repo_id=repo_id,
            token=token,
        )
        logger.info("  [OK]     README.md")
        return True
    except Exception as exc:
        logger.error("  [FAIL]   README.md - %s", exc)
        return False


def main() -> None:
    parser = argparse.ArgumentParser(description="Upload random-noise model checkpoints to Hugging Face.")
    parser.add_argument("--repo-name", default=DEFAULT_REPO, help=f"HF repo name (default: {DEFAULT_REPO})")
    parser.add_argument(
        "--results-dir", default=RESULTS_ROOT, help=f"Results root (default: {RESULTS_ROOT})"
    )
    parser.add_argument(
        "--models",
        nargs="*",
        default=None,
        help="Only upload selected models in the given order",
    )
    parser.add_argument(
        "--display-name",
        action="append",
        default=None,
        help="Optional display mapping in the form model=Display Name. Can be repeated.",
    )
    parser.add_argument("--dry-run", action="store_true", help="Scan and print what would be uploaded")
    parser.add_argument("--no-model-card", action="store_true", help="Skip uploading README.md")
    parser.add_argument(
        "--model-card",
        default=str(DEFAULT_MODEL_CARD),
        help=f"Path to model card README.md (default: {DEFAULT_MODEL_CARD})",
    )
    args = parser.parse_args()

    namespace = os.environ.get("HF_NAMESPACE") or os.environ.get("HF_USERNAME")
    token = os.environ.get("HF_TOKEN")
    selected_models = args.models if args.models else None
    try:
        display_map = build_display_map(args.display_name)
    except ValueError as exc:
        logger.error(str(exc))
        sys.exit(1)
    order_map = build_order_map(selected_models)

    entries = scan_results(args.results_dir, allowed_models=selected_models, order_map=order_map)
    if not entries:
        logger.warning("No matching result folders found in %s", args.results_dir)
        sys.exit(0)
    stats_entries = scan_aggregate_stats(args.results_dir, allowed_models=selected_models, order_map=order_map)

    repo_id = f"{namespace}/{args.repo_name}" if namespace else args.repo_name
    total = len(entries)
    found_pt = sum(1 for e in entries if e["best_pt"])
    found_yaml = sum(1 for e in entries if e["config_yaml"])

    logger.info("Target repo: %s", repo_id)
    logger.info("Selected model order: %s", " ".join(selected_models) if selected_models else "(all discovered models)")
    logger.info("Found %d experiment folders:", total)
    for line in summarize_entries(entries, order_map, display_map):
        logger.info(line)
    logger.info("  - %d have checkpoints/best.pt", found_pt)
    logger.info("  - %d have config.yaml", found_yaml)
    logger.info("  - %d aggregate result file(s) found", len(stats_entries))

    if args.dry_run:
        logger.info("Dry-run mode - files that would be uploaded:")
        for e in entries:
            if e["best_pt"]:
                logger.info("  [upload] %s", hf_path(e, "best.pt"))
            if e["config_yaml"]:
                logger.info("  [upload] %s", hf_path(e, "config.yaml"))
        if not args.no_model_card:
            logger.info("  [upload] README.md (generated from README_models.md + result tables)")
        return

    if not token:
        logger.error("HF_TOKEN environment variable is not set.")
        sys.exit(1)

    api = HfApi()
    logger.info("Creating / ensuring repo: %s", repo_id)
    create_repo(repo_id, token=token, exist_ok=True, private=False)

    try:
        existing = set(api.list_repo_files(repo_id, token=token))
        logger.info("Repo has %d existing file(s); existing files will be skipped.", len(existing))
    except Exception:
        logger.info("Could not list repo files (repo may be empty); uploading all.")
        existing = set()

    uploaded = 0
    skipped = 0
    failed = 0

    if not args.no_model_card:
        model_card_content = build_model_card(
            Path(args.model_card),
            stats_entries,
            repo_id,
            token,
            order_map,
            display_map,
        )
        if upload_model_card(api, repo_id, token, model_card_content):
            uploaded += 1
        else:
            failed += 1

    for e in entries:
        files_to_upload = {}
        if e["best_pt"]:
            files_to_upload["best.pt"] = e["best_pt"]
        if e["config_yaml"]:
            files_to_upload["config.yaml"] = e["config_yaml"]

        for fname, local_path in files_to_upload.items():
            remote = hf_path(e, fname)
            if remote in existing:
                logger.info("  [SKIP]   %s (already exists)", remote)
                skipped += 1
                continue
            try:
                upload_file(
                    path_or_fileobj=local_path,
                    path_in_repo=remote,
                    repo_id=repo_id,
                    token=token,
                )
                logger.info("  [OK]     %s", remote)
                uploaded += 1
            except Exception as exc:
                logger.error("  [FAIL]   %s - %s", remote, exc)
                failed += 1

    logger.info("Done. %d uploaded, %d skipped, %d error(s).", uploaded, skipped, failed)


if __name__ == "__main__":
    main()
