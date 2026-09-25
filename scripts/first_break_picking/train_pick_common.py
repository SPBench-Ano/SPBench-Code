"""Shared training entry for first-break mask segmentation scripts."""

from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Dict, Optional

try:  # works when imported as a package
    from .first_break_data import (
        build_first_break_loaders,
        summarize_first_break_index,
    )
except ImportError:  # works when the training script is executed as a file
    from first_break_data import (  # type: ignore
        build_first_break_loaders,
        summarize_first_break_index,
    )

from model.first_break_picking import build_model
from utils import (
    StepLossLogger,
    TrainingLogger,
    format_final_test_summary,
    write_final_test_metrics,
    barrier_if_distributed,
    build_first_break_metrics,
    build_loss,
    build_optimizer,
    build_scheduler,
    default_config_relpath_for_train_script,
    destroy_distributed,
    evaluate_first_break,
    init_distributed,
    load_checkpoint,
    load_config,
    maybe_save_best_checkpoint,
    maybe_wrap_ddp,
    sampler_set_epoch,
    save_checkpoint,
    set_seed,
    setup_experiment_dir_distributed,
    train_one_epoch_first_break,
    training_device,
    unwrap_ddp,
    visualize_first_break_sample,
)


def parse_args(script_file: str) -> argparse.Namespace:
    default_config = default_config_relpath_for_train_script(script_file)
    default_path = Path(default_config)
    seed42_config = str(default_path.with_name(f"{default_path.stem}_seed42{default_path.suffix}"))
    repo_root = Path(script_file).resolve().parents[2]
    if (repo_root / seed42_config).exists():
        default_config = seed42_config

    parser = argparse.ArgumentParser(
        description="Train first-break picking as binary mask segmentation."
    )
    parser.add_argument(
        "--config",
        type=str,
        default=default_config,
        help="Path to first-break picking YAML config.",
    )
    return parser.parse_args()


def run_training(script_file: str) -> None:
    args = parse_args(script_file)
    cfg = load_config(args.config)

    distributed, rank, local_rank, world_size = init_distributed()
    set_seed(int(cfg["experiment"]["seed"]))
    exp_dir = setup_experiment_dir_distributed(cfg, rank, distributed, base_dir=Path(script_file).resolve().parents[2])
    device = training_device(cfg, distributed=distributed, local_rank=local_rank)

    train_loader, val_loader, test_loader, train_sampler, eval_train_loader, index = build_first_break_loaders(
        cfg,
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
    eval_model = unwrap_ddp(model) if distributed else model
    loss_fn = build_loss(cfg["loss"]).to(device)
    metrics = build_first_break_metrics(cfg.get("metrics", []))
    optimizer = build_optimizer(model, cfg["optim"])
    scheduler = build_scheduler(optimizer, cfg["scheduler"], int(cfg["train"]["epochs"]))

    metric_names = list(metrics.keys())
    logger: Optional[TrainingLogger] = None
    step_loss_logger: Optional[StepLossLogger] = None
    log_dir = exp_dir / cfg["log"].get("log_dir", "logs")
    if rank == 0:
        logger = TrainingLogger(
            log_dir=log_dir,
            loss_keys=["train", "val"],
            metric_keys=[f"train_{m}" for m in metric_names] + [f"val_{m}" for m in metric_names],
            plot_interval=int(cfg["log"].get("plot_interval", 5)),
        )
        step_loss_logger = StepLossLogger(
            log_dir=log_dir,
            flush_interval=int(cfg["train"].get("log_interval", 20)),
        )
        logger.info(summarize_first_break_index(index))
        logger.info(
            f"Model {model_type} | train/val/test patches: "
            f"{len(train_loader.dataset)} / {len(val_loader.dataset)} / {len(test_loader.dataset)}"
        )

    start_epoch = 0
    resume = cfg.get("train", {}).get("resume")
    if resume:
        extras = load_checkpoint(
            resume,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            map_location=device,
        )
        start_epoch = int(extras.get("epoch", -1)) + 1
        if logger is not None:
            logger.info(f"Resumed from {resume} at epoch {start_epoch}.")

    total_epochs = int(cfg["train"]["epochs"])
    eval_interval = int(cfg["train"].get("eval_interval", 1))
    ckpt_interval = int(cfg["train"].get("ckpt_interval", 5))
    vis_interval = int(cfg["train"].get("vis_interval", 5))
    log_step = bool(cfg["train"].get("log_step", False))
    threshold = float(cfg["data"].get("prediction_threshold", 0.5))
    progress_bar = bool(cfg["train"].get("progress_bar", False))
    vis_seed = int(cfg["train"].get("vis_seed", cfg["experiment"]["seed"]))

    best_val_loss = float("inf")
    start_time = time.time()
    for epoch in range(start_epoch, total_epochs):
        epoch_wall_start = time.time()
        sampler_set_epoch(train_sampler, epoch)
        train_stats = train_one_epoch_first_break(
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
            progress_bar=progress_bar and rank == 0,
            step_loss_logger=step_loss_logger,
        )

        val_losses = {"val": float("nan")}
        val_metrics: Dict[str, float] = {}
        train_metrics: Dict[str, float] = {n: float("nan") for n in metric_names}

        if rank == 0 and eval_train_loader is not None:
            _, train_metrics = evaluate_first_break(
                model=eval_model,
                loader=eval_train_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                desc=f"epoch {epoch + 1} eval-train",
                progress_bar=progress_bar,
            )
            if (epoch + 1) % eval_interval == 0:
                val_losses, val_metrics = evaluate_first_break(
                    model=eval_model,
                    loader=val_loader,
                    loss_fn=loss_fn,
                    metrics=metrics,
                    device=device,
                    desc=f"epoch {epoch + 1} val",
                    progress_bar=progress_bar,
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
            epoch_sec = time.time() - epoch_wall_start
            elapsed_min = (time.time() - start_time) / 60.0
            logger.log_epoch(
                epoch=epoch,
                losses={
                    "train": train_stats["train"],
                    "val": val_losses.get("val", float("nan")),
                },
                metrics=metric_row,
                extras={
                    "lr": optimizer.param_groups[0]["lr"],
                    "epoch_sec": epoch_sec,
                    "elapsed_min": elapsed_min,
                },
            )
        if step_loss_logger is not None:
            step_loss_logger.refresh_curves()

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
            visualize_first_break_sample(
                model=eval_model,
                loader=val_loader,
                save_path=exp_dir / "visualizations" / f"epoch_{epoch:04d}.png",
                device=device,
                title=f"First-break {model_type} epoch {epoch}",
                threshold=threshold,
                seed=vis_seed,
            )

        barrier_if_distributed()

    if rank == 0 and logger is not None:
        best_path = exp_dir / "checkpoints" / "best.pt"
        if best_path.exists():
            best_extras = load_checkpoint(
                best_path,
                model=eval_model,
                optimizer=None,
                scheduler=None,
                map_location=device,
            )
            best_epoch = int(best_extras.get("epoch", -1))
            test_losses, test_metrics = evaluate_first_break(
                model=eval_model,
                loader=test_loader,
                loss_fn=loss_fn,
                metrics=metrics,
                device=device,
                desc="final test",
                progress_bar=progress_bar,
                loss_key="test",
            )
            test_loss = test_losses.get("test", float("nan"))
            write_final_test_metrics(
                log_dir / "final_test_metrics.csv",
                best_epoch=best_epoch,
                test_loss=test_loss,
                metrics=test_metrics,
                metric_names=metric_names,
            )
            logger.info(
                format_final_test_summary(
                    best_epoch=best_epoch,
                    test_loss=test_loss,
                    metrics=test_metrics,
                    metric_names=metric_names,
                )
            )
        else:
            logger.info(f"Final test skipped: best checkpoint not found at {best_path}.")

    elapsed = time.time() - start_time
    if logger is not None:
        logger.info(
            f"First-break {model_type} training finished in "
            f"{elapsed:.2f}s ({elapsed / 60:.2f} min)."
        )
    if step_loss_logger is not None:
        step_loss_logger.close()
    if logger is not None:
        logger.close()
    destroy_distributed()


__all__ = ["run_training"]
