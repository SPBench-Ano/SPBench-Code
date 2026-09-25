#!/usr/bin/env bash
# Sequentially run random-noise-suppression train + inference sweeps
# for multiple model families.
set -euo pipefail

# ---------- Configuration ----------
MODEL_LIST=("unet" "dncnn" "res_unet" "atten_unet")
STOP_ON_ERROR=1   # 1 = stop immediately if any train/infer script fails
# -----------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RUN_LOG="${SCRIPT_DIR}/run_all_random_noise_models.log"

timestamp() {
  date -Iseconds
}

log() {
  echo "[$(timestamp)] $*" | tee -a "${RUN_LOG}"
}

run_step() {
  local model="$1"
  local stage="$2"
  local script_path="$3"

  if [[ ! -f "${script_path}" ]]; then
    log "Missing ${stage} script for ${model}: ${script_path}"
    return 1
  fi

  log "Starting ${stage} for ${model}"
  if bash "${script_path}" 2>&1 | tee -a "${RUN_LOG}"; then
    log "Finished ${stage} for ${model}"
    return 0
  fi

  log "FAILED ${stage} for ${model}"
  return 1
}

log "Run started. Models: ${MODEL_LIST[*]}"

for model in "${MODEL_LIST[@]}"; do
  train_script="${SCRIPT_DIR}/train_denoise_${model}.sh"
  infer_script="${SCRIPT_DIR}/inference_denoise_${model}.sh"

  if ! run_step "${model}" "training" "${train_script}"; then
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
    continue
  fi

  if ! run_step "${model}" "inference" "${infer_script}"; then
    if [[ "${STOP_ON_ERROR}" == "1" ]]; then
      exit 1
    fi
  fi
done

log "Run finished."
