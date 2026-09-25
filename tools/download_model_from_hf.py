"""
Download trained model checkpoints (best.pt + config.yaml) from a Hugging Face repository.

The local directory layout mirrors the naming convention expected by
``batch_evaluate.py``::

    <output_dir>/denoise_{model}_base{date}_level{level}_seed{seed}/
        config.yaml
        checkpoints/
            best.pt

Downloads use two-phase parallelism: all config.yaml files are fetched in
parallel first (small, fast), then all best.pt checkpoints are downloaded
in parallel (large, benefits from concurrent connections).

Usage:
    export HF_TOKEN="your_hf_token"   # only needed for private / gated repos
    python tools/download_model_from_hf.py --models unet res_unet

Choose from: unet, res_unet, dncnn, atten_unet, enhanced_unet, ddpm,
              pix2pix, sanet, physics

Optional:
    --repo-id ID          HF repo ID (default: SeisProcBench/SPBench-Model)
    --output-dir PATH     Local output directory (default: ./downloaded_models)
    --noise-levels L [L]  Only download specific noise levels (default: all)
    --seeds S [S ...]     Only download specific seeds (default: all)
    --num-workers N       Parallel download workers (default: 4)
    --dry-run             List what would be downloaded without downloading
    --models M [M ...]    Only download the listed model(s); default = all
    --force               Re-download even if local files already exist
"""

import argparse
import logging
import os
import re
import shutil
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed

import yaml
from huggingface_hub import HfApi, hf_hub_download

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)

DEFAULT_REPO_ID = "SeisProcBench/SPBench-Model"
DEFAULT_OUTPUT_DIR = "downloaded_models"
DEFAULT_NUM_WORKERS = 4

# Remote path pattern: models/{model}/level{level}_seed{seed}/{filename}
_REMOTE_PATH_RE = re.compile(
    r"^models/(?P<model>[^/]+)/level(?P<level>[\d.]+)_seed(?P<seed>\d+)/(?P<file>.+)$"
)

MODEL_DISPLAY = {
    "unet": "UNet",
    "res_unet": "ResUNet",
    "dncnn": "DnCNN",
    "atten_unet": "Attention UNet",
    "enhanced_atten_unet": "Enhanced Atten-UNet",
    "sanet": "SANet",
    "physics_unet": "Physics CNN",
    "pix2pix": "Pix2Pix cGAN",
    "ddpm": "DDPM cDDPM",
}

# User-facing name → internal folder-pattern key
_MODEL_ALIASES = {
    "enhanced_unet": "enhanced_atten_unet",
    "physics": "physics_unet",
}

_ALL_MODELS = sorted(
    ["unet", "res_unet", "dncnn", "atten_unet", "enhanced_atten_unet",
     "ddpm", "pix2pix", "sanet", "physics_unet"]
)

_VALID_MODEL_NAMES = sorted(list(_MODEL_ALIASES.keys()) + _ALL_MODELS)


def _resolve_models(requested):
    """Resolve user-facing model names to internal folder-pattern keys."""
    if not requested:
        return _ALL_MODELS
    resolved = []
    for name in requested:
        name = name.strip().lower()
        if name in _MODEL_ALIASES:
            name = _MODEL_ALIASES[name]
        if name not in _ALL_MODELS:
            raise ValueError(
                f"Unknown model {name!r}. Valid choices: {', '.join(_VALID_MODEL_NAMES)}"
            )
        resolved.append(name)
    return resolved


def list_remote_entries(repo_id: str, token: str | None) -> list[dict]:
    """Fetch the file listing from the HF repo and parse into structured entries.

    Returns a list of dicts with keys: ``model``, ``level``, ``seed``.
    """
    api = HfApi()
    try:
        files = api.list_repo_files(repo_id, token=token)
    except Exception as exc:
        logger.error("Failed to list repo files for %s: %s", repo_id, exc)
        sys.exit(1)

    entries: list[dict] = []
    seen: set[tuple] = set()

    for path in files:
        m = _REMOTE_PATH_RE.match(path)
        if not m:
            continue
        key = (m.group("model"), m.group("level"), m.group("seed"))
        if key not in seen:
            entries.append({
                "model": m.group("model"),
                "level": m.group("level"),
                "seed": m.group("seed"),
            })
            seen.add(key)

    entries.sort(key=lambda x: (x["model"], float(x["level"]), int(x["seed"])))
    return entries


def _build_exp_dir_name(exp_name: str, level: str, seed: str) -> str:
    """Append ``_level{level}_seed{seed}`` to *exp_name* if not already present."""
    if f"_level{level}" in exp_name:
        return exp_name
    return f"{exp_name}_level{level}_seed{seed}"


# ---------------------------------------------------------------------------
# Phase-1 worker: download a single config.yaml and return experiment metadata
# ---------------------------------------------------------------------------
def _fetch_config_worker(
    repo_id: str, model: str, level: str, seed: str, token: str | None,
) -> tuple[str, str, str, str, str] | tuple[None, str]:
    """Download config.yaml, return ``(model, level, seed, exp_dir_name, cache_path)``.

    On failure returns ``(None, error_message)``.
    """
    remote = f"models/{model}/level{level}_seed{seed}/config.yaml"
    try:
        cache_path = hf_hub_download(
            repo_id=repo_id, filename=remote, token=token,
        )
        with open(cache_path, "r") as f:
            cfg = yaml.safe_load(f)
        exp_name = cfg.get("experiment", {}).get("name", f"denoise_{model}_base0000")
        exp_dir_name = _build_exp_dir_name(exp_name, level, seed)
        return model, level, seed, exp_dir_name, cache_path
    except Exception as exc:
        return None, f"{remote}: {exc}"


# ---------------------------------------------------------------------------
# Phase-2 worker: download best.pt and place into target checkpoints/ dir
# ---------------------------------------------------------------------------
def _fetch_ckpt_worker(
    repo_id: str,
    model: str,
    level: str,
    seed: str,
    exp_dir_name: str,
    output_dir: str,
    token: str | None,
    force: bool,
) -> tuple[bool, str]:
    """Download best.pt → ``<output_dir>/<exp_dir_name>/checkpoints/best.pt``.

    Returns ``(True, "ok"|"skipped")`` or ``(False, error_message)``.
    """
    remote = f"models/{model}/level{level}_seed{seed}/best.pt"
    target_dir = os.path.join(output_dir, exp_dir_name)
    target_ckpt_dir = os.path.join(target_dir, "checkpoints")
    target_ckpt = os.path.join(target_ckpt_dir, "best.pt")

    if os.path.isfile(target_ckpt) and not force:
        return True, "skipped"

    try:
        cache_path = hf_hub_download(
            repo_id=repo_id, filename=remote, token=token,
        )
        os.makedirs(target_ckpt_dir, exist_ok=True)
        shutil.copy2(cache_path, target_ckpt)
        return True, "ok"
    except Exception as exc:
        return False, f"{remote}: {exc}"


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Download model checkpoints from a Hugging Face repository."
    )
    parser.add_argument(
        "--repo-id", default=DEFAULT_REPO_ID,
        help=f"HF repo ID (default: {DEFAULT_REPO_ID})",
    )
    parser.add_argument(
        "--output-dir", default=DEFAULT_OUTPUT_DIR,
        help=f"Local output directory (default: {DEFAULT_OUTPUT_DIR})",
    )
    parser.add_argument(
        "--models", nargs="*", metavar="MODEL", default=None,
        help="Only download the listed model(s).  Valid: %s.  Default: all."
             % ", ".join(_VALID_MODEL_NAMES),
    )
    parser.add_argument(
        "--noise-levels", nargs="*", metavar="LEVEL", default=None, type=str,
        help="Only download the listed noise level(s), e.g. 1.0 3.0.  Default: all.",
    )
    parser.add_argument(
        "--seeds", nargs="*", metavar="SEED", default=None, type=str,
        help="Only download the listed seed(s), e.g. 42 43 44.  Default: all.",
    )
    parser.add_argument(
        "--num-workers", type=int, default=DEFAULT_NUM_WORKERS,
        help=f"Number of parallel download workers (default: {DEFAULT_NUM_WORKERS}).",
    )
    parser.add_argument(
        "--dry-run", action="store_true",
        help="Scan and print what would be downloaded without downloading.",
    )
    parser.add_argument(
        "--force", action="store_true",
        help="Re-download even if the local file already exists.",
    )
    args = parser.parse_args()

    token = os.environ.get("HF_TOKEN")

    # --- discover remote entries ----------------------------------------------
    logger.info("Listing files in repo: %s", args.repo_id)
    entries = list_remote_entries(args.repo_id, token)

    if not entries:
        logger.warning("No model entries found in repo %s.", args.repo_id)
        sys.exit(0)

    # --- filter by model ------------------------------------------------------
    selected = _resolve_models(args.models)
    if args.models:
        entries = [e for e in entries if e["model"] in selected]
        if not entries:
            logger.warning(
                "No entries matched models: %s.  Available: %s",
                ", ".join(selected),
                ", ".join(sorted(
                    {e["model"] for e in list_remote_entries(args.repo_id, token)}
                )),
            )
            sys.exit(0)
        logger.info(
            "Selected models: %s → %d entry(s)", ", ".join(selected), len(entries)
        )

    # --- filter by noise level ------------------------------------------------
    if args.noise_levels:
        entries = [e for e in entries if e["level"] in args.noise_levels]
        if not entries:
            logger.warning(
                "No entries matched noise levels: %s", ", ".join(args.noise_levels)
            )
            sys.exit(0)
        logger.info("Filtered by level → %d entry(s)", len(entries))

    # --- filter by seed -------------------------------------------------------
    if args.seeds:
        entries = [e for e in entries if e["seed"] in args.seeds]
        if not entries:
            logger.warning("No entries matched seeds: %s", ", ".join(args.seeds))
            sys.exit(0)
        logger.info("Filtered by seed → %d entry(s)", len(entries))

    # --- summarize ------------------------------------------------------------
    models = sorted({e["model"] for e in entries})
    levels = sorted({e["level"] for e in entries}, key=float)
    seeds = sorted({e["seed"] for e in entries}, key=int)

    logger.info(
        "Summary: %d entry(s) across %d model(s), %d level(s), %d seed(s).",
        len(entries), len(models), len(levels), len(seeds),
    )
    for m in models:
        count = sum(1 for e in entries if e["model"] == m)
        logger.info("  %-22s  %d checkpoint(s)", MODEL_DISPLAY.get(m, m), count)

    if args.dry_run:
        logger.info("Dry-run mode — directories that would be created:")
        for e in entries:
            guessed = f"denoise_{e['model']}_baseXXXX_level{e['level']}_seed{e['seed']}"
            logger.info("  %s/", guessed)
            logger.info("    ├── config.yaml")
            logger.info("    └── checkpoints/best.pt")
        return

    output_dir = os.path.abspath(args.output_dir)
    n_workers = max(1, args.num_workers)
    logger.info("Downloading to: %s  (workers: %d)", output_dir, n_workers)

    # =========================================================================
    # Phase 1 — parallel download all config.yaml files (small, fast)
    # =========================================================================
    logger.info("=== Phase 1: downloading config.yaml (%d files) ===", len(entries))
    resolved: list[dict] = []   # entries augmented with exp_dir_name
    config_fail = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_entry = {
            pool.submit(
                _fetch_config_worker, args.repo_id,
                e["model"], e["level"], e["seed"], token,
            ): e
            for e in entries
        }
        for future in as_completed(future_to_entry):
            result = future.result()
            if result[0] is None:
                logger.error("  [FAIL]   %s", result[1])
                config_fail += 1
                continue
            model, level, seed, exp_dir_name, config_cache = result
            # Save config.yaml to target
            target_dir = os.path.join(output_dir, exp_dir_name)
            target_config = os.path.join(target_dir, "config.yaml")
            if os.path.isfile(target_config) and not args.force:
                logger.info("  [SKIP]   %s/config.yaml (exists)", exp_dir_name)
            else:
                os.makedirs(target_dir, exist_ok=True)
                shutil.copy2(config_cache, target_config)
                logger.info("  [OK]     %s/config.yaml", exp_dir_name)
            resolved.append({
                "model": model, "level": level, "seed": seed,
                "exp_dir_name": exp_dir_name,
            })

    if config_fail:
        logger.warning("%d config.yaml download(s) failed — skipping those entries.", config_fail)
    if not resolved:
        logger.error("No config.yaml files downloaded successfully.")
        sys.exit(1)

    # =========================================================================
    # Phase 2 — parallel download all best.pt files (large, benefits from concurrency)
    # =========================================================================
    logger.info("=== Phase 2: downloading best.pt (%d files) ===", len(resolved))
    ok = 0
    skipped = 0
    ckpt_fail = 0

    with ThreadPoolExecutor(max_workers=n_workers) as pool:
        future_to_item = {
            pool.submit(
                _fetch_ckpt_worker, args.repo_id,
                r["model"], r["level"], r["seed"], r["exp_dir_name"],
                output_dir, token, args.force,
            ): r
            for r in resolved
        }
        for future in as_completed(future_to_item):
            item = future_to_item[future]
            success, status = future.result()
            label = f"{item['exp_dir_name']}/checkpoints/best.pt"
            if not success:
                logger.error("  [FAIL]   %s", label)
                ckpt_fail += 1
            elif status == "skipped":
                logger.info("  [SKIP]   %s (exists)", label)
                skipped += 1
                ok += 1
            else:
                logger.info("  [OK]     %s", label)
                ok += 1

    logger.info(
        "Done. %d succeeded, %d skipped, %d config-fail, %d ckpt-fail.",
        ok, skipped, config_fail, ckpt_fail,
    )


if __name__ == "__main__":
    main()
