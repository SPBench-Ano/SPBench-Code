#!/usr/bin/env bash
# Run random-noise-suppression inference over multiple noise kinds / SNR levels / seeds,
# then aggregate mean/std across seeds.
set -euo pipefail

unset DISPLAY
export MPLBACKEND=Agg

# ---------- Configuration ----------
DEVICE="cuda:0"
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)
N=3                         # Number of runs / seeds
START_SEED=42               # Seeds: START_SEED, START_SEED+1, ...
N_VIZ_SHOTS=5
SAVE_NPY=0                  # 1 = save .npy outputs, 0 = skip
CHECKPOINT_NAME="best.pt"   # e.g. "best.pt" or "epoch_0049.pt"
# -----------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/random_noise_suppression/denoise_unet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/random_noise_suppression/inference_denoise_unet.py"

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi
if ! command -v python >/dev/null 2>&1; then
  echo "python not found in current environment." >&2
  exit 1
fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
OUTPUT_ROOT="$(grep -m1 -E '^[[:space:]]*output_dir:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*output_dir:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"

if [[ -z "${NAME_BASE}" ]]; then
  echo "Could not parse experiment.name from ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ -z "${OUTPUT_ROOT}" ]]; then
  echo "Could not parse experiment.output_dir from ${BASE_CONFIG}" >&2
  exit 1
fi

for noise_kind in "${NOISE_KIND_LIST[@]}"; do
  for snr_db in "${SNR_LIST[@]}"; do
    if (( snr_db < 0 )); then
      snr_tag="neg$((-snr_db))"
    else
      snr_tag="${snr_db}"
    fi

    summary_paths=()
    seeds=()

    for ((i = 0; i < N; i++)); do
      run_seed=$((START_SEED + i))
      run_name="${NAME_BASE}_${noise_kind}_snr${snr_tag}_seed${run_seed}"
      checkpoint_path="${OUTPUT_ROOT}/${run_name}/checkpoints/${CHECKPOINT_NAME}"
      infer_output_dir="${OUTPUT_ROOT}/${run_name}/inference"
      log_path="${infer_output_dir}/inference.log"

      if [[ ! -f "${checkpoint_path}" ]]; then
        echo "[$(date -Iseconds)] Skip missing checkpoint: ${checkpoint_path}" >&2
        continue
      fi

      CMD=(
        python
        "${PY_SCRIPT}"
        --config "${BASE_CONFIG}"
        --checkpoint "${checkpoint_path}"
        --output-dir "${infer_output_dir}"
        --device "${DEVICE}"
        --n-viz-shots "${N_VIZ_SHOTS}"
        --seed "${run_seed}"
        --noise-kind "${noise_kind}"
        --snr-db "${snr_db}"
      )

      if [[ "${SAVE_NPY}" == "1" ]]; then
        CMD+=(--save-npy)
      fi

      mkdir -p "${infer_output_dir}"
      {
        echo "[$(date -Iseconds)] Starting inference"
        echo "  run_name: ${run_name}"
        echo "  config: ${BASE_CONFIG}"
        echo "  checkpoint: ${checkpoint_path}"
        echo "  output_dir: ${infer_output_dir}"
        echo "  noise_kind: ${noise_kind}"
        echo "  snr_db: ${snr_db}"
        echo "  seed: ${run_seed}"
      } | tee -a "${log_path}"

      cd "${REPO_ROOT}"
      "${CMD[@]}" 2>&1 | tee -a "${log_path}"

      summary_path="${infer_output_dir}/metrics_summary.json"
      if [[ -f "${summary_path}" ]]; then
        summary_paths+=("${summary_path}")
        seeds+=("${run_seed}")
      fi
    done

    if (( ${#summary_paths[@]} == 0 )); then
      echo "[$(date -Iseconds)] No summaries found for ${noise_kind} snr=${snr_db}" >&2
      continue
    fi

    aggregate_dir="${OUTPUT_ROOT}/${NAME_BASE}_${noise_kind}_snr${snr_tag}_seed_stats"
    mkdir -p "${aggregate_dir}"
    aggregate_json="${aggregate_dir}/metrics_summary_mean_std.json"
    aggregate_log="${aggregate_dir}/aggregate.log"

    {
      echo "[$(date -Iseconds)] Aggregating summaries"
      echo "  output: ${aggregate_json}"
      echo "  noise_kind: ${noise_kind}"
      echo "  snr_db: ${snr_db}"
      echo "  seeds: ${seeds[*]}"
    } | tee -a "${aggregate_log}"

    python - "${aggregate_json}" "${noise_kind}" "${snr_db}" "${NAME_BASE}" "${seeds[@]}" -- "${summary_paths[@]}" <<'PY' 2>&1 | tee -a "${aggregate_log}"
import json
import statistics
import sys
from pathlib import Path

args = sys.argv[1:]
out_path = Path(args[0])
noise_kind = args[1]
snr_db = float(args[2])
experiment = args[3]
sep = args.index("--")
seed_args = args[4:sep]
summary_args = args[sep + 1 :]
seeds = [int(s) for s in seed_args]

summaries = [json.loads(Path(p).read_text(encoding="utf-8")) for p in summary_args]
groups = ("noisy", "denoised", "delta")
result = {
    "experiment": experiment,
    "noise_kind": noise_kind,
    "snr_db": snr_db,
    "seeds": seeds,
    "num_runs": len(seeds),
}

for group in groups:
    metrics = sorted(summaries[0][group].keys())
    result[group] = {}
    for metric in metrics:
        vals = [float(s[group][metric]) for s in summaries]
        mean = sum(vals) / len(vals)
        std = statistics.pstdev(vals) if len(vals) > 1 else 0.0
        result[group][metric] = {
            "mean": round(mean, 6),
            "std": round(std, 6),
        }

out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
print(f"Saved aggregate summary to: {out_path}")
print(f"Seeds: {seeds}")
PY
  done
done

echo "[$(date -Iseconds)] Inference sweep finished."
