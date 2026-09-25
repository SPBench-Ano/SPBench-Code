"""SEG-Y interpolation: uniform trace-missing + supervised training (YAML model; multi-GPU via ``torchrun`` + DDP).

CUDA_VISIBLE_DEVICES=6,7 torchrun --nproc_per_node=2 \
    scripts/interpolation/train_interpolation_unet.py \
    --config configs/interpolation/interpolation_unet.yaml
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import numpy as np
import torch

# Bootstrap repo root into sys.path BEFORE importing utils/model. Walks up from
# this file looking for a directory that contains both ``model/`` and ``utils/``.
_REPO_ROOT = next(
    (p for p in Path(__file__).resolve().parents
     if (p / "model").is_dir() and (p / "utils").is_dir()),
    None,
)
if _REPO_ROOT is None:
    raise RuntimeError(
        "Cannot find repo root (a directory containing both ``model/`` and ``utils/``)."
    )
if str(_REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(_REPO_ROOT))

from model.interpolation import build_model  # noqa: E402
from tools.patching import patchify_uniform  # noqa: E402
from tools.array_io import load_volume  # noqa: E402
from tools.preprocessing import mask_traces, normalize, spherical_divergence_correction  # noqa: E402
from utils import (  # noqa: E402
    TrainingLogger,
    build_loss,
    build_loaders,
    build_metrics,
    build_optimizer,
    build_scheduler,
    build_shot_split_loaders,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate,
    init_distributed,
    load_config,
    maybe_wrap_ddp,
    sampler_set_epoch,
    maybe_save_best_checkpoint,
    save_checkpoint,
    set_seed,
    setup_experiment_dir_distributed,
    train_one_epoch,
    training_device,
    visualize_random_sample,
)


def _preprocess_shots(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load volume, preprocess, and return ``(input_shots, target_shots, per_shot_ffid)``."""
    prep = cfg["preprocess"]

    data_cfg = None
    for key in ("segy", "npy", "mat"):
        if key in cfg["data"]:
            data_cfg = cfg["data"][key]
            break
    if data_cfg is None:
        raise ValueError(
            "No data source found in config (expected data.segy, data.npy, or data.mat)."
        )

    shots = load_volume(data_cfg)

    if prep.get("max_shots") is not None:
        shots = shots[: int(prep["max_shots"])]

    skip = set(prep.get("skip", []))

    if "spherical_divergence_correction" not in skip:
        shots = spherical_divergence_correction(
            shots,
            dt=float(prep["dt"]),
            power=float(prep.get("spherical_power", 1.2)),
            t0=float(prep.get("t0", 0.0)),
        )

    if "normalize" not in skip:
        shots, _ = normalize(
            shots,
            mode=str(prep.get("normalize_mode", "max_abs")),
            per=str(prep.get("normalize_scope", "global")),
        )

    if "mask_traces" not in skip:
        mask_mode = str(prep.get("mask_mode", "uniform"))
        mask_ratio = float(prep.get("mask_ratio", 0.5))
        mask_kwargs: Dict[str, Any] = {"mode": mask_mode, "ratio": mask_ratio}
        if mask_mode == "uniform":
            mask_kwargs["uniform_stride"] = int(prep.get("uniform_stride", 2))
        masked, _ = mask_traces(shots, **mask_kwargs)
    else:
        masked = shots

    # Extract per-shot FFID for shot-level splitting.
    if data_cfg.get("path", "").lower().endswith((".sgy", ".segy")):
        from tools.segy_read import read_regular_shots

        _, headers = read_regular_shots(
            data_cfg["path"],
            traces_per_shot=int(data_cfg.get("traces_per_shot", 201)),
            time_downsample=int(data_cfg.get("time_downsample", 1)),
            return_headers=True,
        )
        per_shot_ffid = headers["FieldRecord"][:, 0]
    else:
        per_shot_ffid = np.arange(shots.shape[0])

    return masked, shots, per_shot_ffid


def _patchify_pairs(
    input_shots: np.ndarray, target_shots: np.ndarray, cfg: Dict[str, Any]
) -> Tuple[np.ndarray, np.ndarray]:
    """Patchify given shot subsets."""
    prep = cfg["preprocess"]
    patch_t = int(prep.get("patch_time", 256))
    patch_x = int(prep.get("patch_trace", 128))
    overlap = float(prep.get("patch_overlap", 0.5))

    target_patches, _ = patchify_uniform(
        target_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    input_patches, _ = patchify_uniform(
        input_shots, patch_size=(patch_x, patch_t), overlap=overlap, output_ndim=4
    )
    return input_patches.astype(np.float32), target_patches.astype(np.float32)


def _build_patch_pairs(cfg: Dict[str, Any]) -> Tuple[np.ndarray, np.ndarray]:
    """Backward-compatible full pipeline."""
    inp, tgt, _ = _preprocess_shots(cfg)
    return _patchify_pairs(inp, tgt, cfg)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Train interpolation from a single SEG-Y volume with uniform trace masking. "
            "Multi-GPU: torchrun --nproc_per_node=N ..."
        )
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config_relpath_for_train_script(__file__),
        help="Path to interpolation config.",
    )
    parser.add_argument(
        "--mask-mode",
        type=str,
        default="uniform",
        choices=["uniform", "random", "continuous"],
        help="Trace masking mode.",
    )
    parser.add_argument(
        "--mask-ratio",
        type=float,
        default=0.5,
        help="Trace missing ratio in (0, 1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = load_config(args.config)

    # Inject CLI mask args into preprocess config so _build_patch_pairs can read them.
    cfg.setdefault("preprocess", {})
    cfg["preprocess"]["mask_mode"] = args.mask_mode
    cfg["preprocess"]["mask_ratio"] = args.mask_ratio

    ratio_pct = int(round(args.mask_ratio * 100))
    cfg["experiment"]["name"] = f"{cfg['experiment']['name']}_{args.mask_mode}_miss{ratio_pct}"

    distributed, rank, local_rank, world_size = init_distributed()

    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=_REPO_ROOT)
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    if "shot_split" in cfg.get("data", {}):
        train_loader, val_loader, train_sampler, eval_train_loader = build_shot_split_loaders(
            cfg,
            preprocess_fn=_preprocess_shots,
            patchify_fn=_patchify_pairs,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
            test_set_dir=exp_dir / "test_set",
        )
    else:
        train_loader, val_loader, train_sampler, eval_train_loader = build_loaders(
            cfg,
            build_patch_pairs_fn=_build_patch_pairs,
            rank=rank,
            world_size=world_size,
            distributed=distributed,
        )

    model = build_model(cfg["model"]).to(device)
    model = maybe_wrap_ddp(
        model,
        distributed=distributed,
        device=device,
        local_rank=local_rank,
    )
    model_type = str(cfg["model"]["type"])
    loss_fn = build_loss(cfg["loss"]).to(device)
    metrics = build_metrics(cfg.get("metrics", []))
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    if rank == 0:
        logger = TrainingLogger(
            log_dir=exp_dir / cfg["log"].get("log_dir", "logs"),
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
    if logger is not None:
        logger.info(
            f"Model {model_type} | train/val patches: {len(train_loader.dataset)} / {len(val_loader.dataset)}"
        )

    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(total_epochs):
        sampler_set_epoch(train_sampler, epoch)
        train_stats = train_one_epoch(
            model=model,
            loader=train_loader,
            loss_fn=loss_fn,
            optimizer=optimizer,
            device=device,
            epoch=epoch,
            scheduler=scheduler,
            grad_clip=cfg["train"].get("grad_clip"),
            log_interval=int(cfg["train"].get("log_interval", 20)),
            logger=logger if log_step else None,
        )
        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate(
                model=model,
                loader=eval_train_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate(
                    model=model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device,
                )
                best_val_loss = maybe_save_best_checkpoint(
                    exp_dir / "checkpoints" / "best.pt",
                    model=model,
                    optimizer=optimizer,
                    scheduler=scheduler,
                    epoch=epoch,
                    val_loss=val_losses["val"],
                    best_val_loss=best_val_loss,
                    extras={"config": cfg},
                    logger=logger,
                )

        metric_row: Dict[str, float] = {}
        for name in metric_names:
            metric_row[f"train_{name}"] = train_metrics.get(name, float("nan"))
            metric_row[f"val_{name}"] = val_metrics.get(name, float("nan"))

        if logger is not None:
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train": train_stats["train"],
                    "val": val_losses.get("val", float("nan")),
                },
                metrics=metric_row,
                extras={"lr": optimizer.param_groups[0]["lr"]},
            )

        if rank == 0 and (epoch + 1) % ckpt_interval == 0:
            save_checkpoint(
                exp_dir / "checkpoints" / f"epoch_{epoch:04d}.pt",
                model=model,
                optimizer=optimizer,
                scheduler=scheduler,
                epoch=epoch,
                extras={"config": cfg},
            )

        if rank == 0 and (epoch + 1) % vis_interval == 0:
            visualize_random_sample(
                model=model,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"Interpolation {model_type} epoch {epoch}",
                seed=None,
            )

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(f"Interpolation {model_type} training finished in {elapsed:.2f}s ({elapsed/60:.2f} min).")
        logger.close()
    destroy_distributed()


if __name__ == "__main__":
    main()
