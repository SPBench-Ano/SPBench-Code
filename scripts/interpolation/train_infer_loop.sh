#!/usr/bin/env bash
# Train + Inference pipeline: loops over mask modes, mask ratios, and seeds.
# For each combination: trains the model, then runs inference on the best checkpoint.
#
# The training script appends ``_{mask_mode}_miss{ratio_pct}`` to the experiment
# name, so results land in separate directories per configuration.

set -euo pipefail

# ---------- Configuration (edit here) ----------
CUDA_VISIBLE_DEVICES="1"
NPROC_PER_NODE=1

# Multi-mask loops
MASK_MODES=( "random" "continuous" “uniform”)
MASK_RATIOS=("0.3" "0.5" "0.7")

# Seed loop
N_SEEDS=3
START_SEED=42

# Paths
BASE_CONFIG="configs/interpolation/interpolation_unet.yaml"
TRAIN_PY="scripts/interpolation/train_interpolation_unet.py"
INFER_PY="scripts/interpolation/inference_interpolation.py"

# Inference settings
INFER_DEVICE="cuda:1"

# MASTER_PORT: set to a fixed port, or "auto" to pick a free port each run
MASTER_PORT="auto"
TORCHRUN_EXTRA=""
# ----------------------------------------------

_pick_port() {
  python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

export CUDA_VISIBLE_DEVICES

for f in "${BASE_CONFIG}" "${TRAIN_PY}" "${INFER_PY}"; do
  full="${REPO_ROOT}/${f}"
  if [[ ! -f "${full}" ]]; then
    echo "File not found: ${full}" >&2
    exit 1
  fi
done

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${REPO_ROOT}/${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
if [[ -z "${NAME_BASE}" ]]; then
  echo "Could not parse experiment.name from ${BASE_CONFIG}" >&2
  exit 1
fi

tmpcfg="$(mktemp)"
cleanup() { rm -f "${tmpcfg}"; }
trap cleanup EXIT

TOTAL_RUNS=$(( ${#MASK_MODES[@]} * ${#MASK_RATIOS[@]} * N_SEEDS ))
run_idx=0

for mask_mode in "${MASK_MODES[@]}"; do
  for mask_ratio in "${MASK_RATIOS[@]}"; do
    ratio_pct=$(echo "${mask_ratio}" | awk '{printf "%d", $1*100}')

    for ((i = 0; i < N_SEEDS; i++)); do
      seed=$((START_SEED + i))
      run_idx=$((run_idx + 1))

      # sed only sets base+seed; Python appends _<mode>_miss<ratio>
      run_name_sed="${NAME_BASE}_seed${seed}"
      run_name_full="${NAME_BASE}_seed${seed}_${mask_mode}_miss${ratio_pct}"

      echo "============================================"
      echo "[$(date -Iseconds)] [${run_idx}/${TOTAL_RUNS}] mode=${mask_mode} ratio=${mask_ratio} seed=${seed}"
      echo "  Experiment: ${run_name_full}"
      echo "============================================"

      # --- Prepare temp config ---
      sed -E \
        -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
        -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name_sed}"'/' \
        "${REPO_ROOT}/${BASE_CONFIG}" >"${tmpcfg}"

      # --- Train ---
      echo "[$(date -Iseconds)] Training..."
      cd "${REPO_ROOT}"
      if [[ "${MASTER_PORT}" == "auto" ]]; then
        master_port=$(_pick_port)
      else
        master_port="${MASTER_PORT}"
      fi
      # shellcheck disable=SC2086
      torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" \
        --master_port="${master_port}" \
        "${TRAIN_PY}" --config "${tmpcfg}" \
        --mask-mode "${mask_mode}" --mask-ratio "${mask_ratio}"

      # --- Find latest checkpoint ---
      ckpt_dir="${REPO_ROOT}/results/${run_name_full}/checkpoints"
      latest_ckpt=$(ls -t "${ckpt_dir}"/epoch_*.pt 2>/dev/null | head -1)
      if [[ -z "${latest_ckpt}" ]]; then
        echo "[$(date -Iseconds)] WARNING: No checkpoint found in ${ckpt_dir}, skipping inference." >&2
        continue
      fi
      echo "[$(date -Iseconds)] Using checkpoint: ${latest_ckpt}"

      # --- Infer (matching training ratio) ---
      infer_out="${REPO_ROOT}/results/${run_name_full}/inference"
      # shellcheck disable=SC2086
      python "${REPO_ROOT}/${INFER_PY}" \
        --config "${tmpcfg}" \
        --checkpoint "${latest_ckpt}" \
        --output-dir "${infer_out}" \
        --mask-mode "${mask_mode}" --mask-ratio "${mask_ratio}" \
        --device "${INFER_DEVICE}"

      # --- Infer (extra: +0.1 missing ratio, tests generalization) ---
      ratio_extra=$(awk "BEGIN {printf \"%.1f\", ${mask_ratio}+0.1}")
      ratio_extra_pct=$(echo "${ratio_extra}" | awk '{printf "%d", $1*100}')
      infer_out_extra="${REPO_ROOT}/results/${run_name_full}/inference_ratio${ratio_extra_pct}"
      # shellcheck disable=SC2086
      python "${REPO_ROOT}/${INFER_PY}" \
        --config "${tmpcfg}" \
        --checkpoint "${latest_ckpt}" \
        --output-dir "${infer_out_extra}" \
        --mask-mode "${mask_mode}" --mask-ratio "${ratio_extra}" \
        --device "${INFER_DEVICE}"

      echo "[$(date -Iseconds)] Done: ${run_name_full}"
      echo ""
    done
  done
done

echo "[$(date -Iseconds)] All ${TOTAL_RUNS} runs complete."
