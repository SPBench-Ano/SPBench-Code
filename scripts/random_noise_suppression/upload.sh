#!/usr/bin/env bash
# Upload random-noise-suppression checkpoints + configs to a Hugging Face model repo.
# The script loops by model family first so upload order is easy to adjust.
set -euo pipefail

# ---------- User-tunable configuration ----------
HF_REPO_URL="https://huggingface.co/SeisProcBench/SPBench-Model"
HF_REPO_ID="SeisProcBench/SPBench-Model"

# Local training results root on the Linux training machine.
RESULTS_ROOT="/data/pcdisk0/data/results/random_noise"

# Where to clone / reuse the Hugging Face git repo locally.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
HF_LOCAL_REPO="${SCRIPT_DIR}/hf_random-noise-attenuation"

# Upload order is controlled here.
MODEL_LIST=("unet" "dncnn" "res_unet" "atten_unet" "SCRN")

# 1 = print actions only, no file copy / git commit / git push.
DRY_RUN=1

# 1 = push after commit, 0 = leave local commit only.
PUSH=0

# Commit once after staging all copied files.
COMMIT_MESSAGE="Add random-noise attenuation checkpoints and configs"
# -----------------------------------------------

timestamp() {
  date -Iseconds
}

log() {
  echo "[$(timestamp)] $*"
}

require_cmd() {
  local cmd="$1"
  if ! command -v "${cmd}" >/dev/null 2>&1; then
    echo "Required command not found: ${cmd}" >&2
    exit 1
  fi
}

run_cmd() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] $*"
  else
    "$@"
  fi
}

ensure_repo() {
  require_cmd git

  if [[ -d "${HF_LOCAL_REPO}/.git" ]]; then
    log "Using existing Hugging Face repo clone: ${HF_LOCAL_REPO}"
    return 0
  fi

  log "Cloning Hugging Face repo: ${HF_REPO_URL}"
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] git clone ${HF_REPO_URL} ${HF_LOCAL_REPO}"
  else
    git clone "${HF_REPO_URL}" "${HF_LOCAL_REPO}"
  fi
}

copy_one_run() {
  local model="$1"
  local run_dir="$2"
  local run_name
  local suffix
  local dst_rel
  local dst_dir
  local src_ckpt
  local src_cfg

  run_name="$(basename "${run_dir}")"
  suffix="${run_name#random_noise_${model}_base_}"
  dst_rel="models/${model}/${suffix}"
  dst_dir="${HF_LOCAL_REPO}/${dst_rel}"

  src_ckpt="${run_dir}/checkpoints/best.pt"
  src_cfg="${run_dir}/config.yaml"

  if [[ ! -f "${src_ckpt}" ]]; then
    log "Skip missing checkpoint: ${src_ckpt}"
    return 0
  fi
  if [[ ! -f "${src_cfg}" ]]; then
    log "Skip missing config: ${src_cfg}"
    return 0
  fi

  log "Upload mapping:"
  log "  model: ${model}"
  log "  source run: ${run_name}"
  log "  checkpoint: ${src_ckpt} -> ${dst_rel}/best.pt"
  log "  config: ${src_cfg} -> ${dst_rel}/config.yaml"

  run_cmd mkdir -p "${dst_dir}"
  run_cmd cp "${src_ckpt}" "${dst_dir}/best.pt"
  run_cmd cp "${src_cfg}" "${dst_dir}/config.yaml"
}

upload_model_group() {
  local model="$1"
  local pattern="random_noise_${model}_base_*_seed*"
  local found=0
  local run_dir

  log "Scanning model group: ${model}"
  shopt -s nullglob
  for run_dir in "${RESULTS_ROOT}"/${pattern}; do
    if [[ -d "${run_dir}" ]]; then
      found=1
      copy_one_run "${model}" "${run_dir}"
    fi
  done
  shopt -u nullglob

  if [[ "${found}" == "0" ]]; then
    log "No runs found for model=${model} under ${RESULTS_ROOT}"
  fi
}

stage_commit_push() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[DRY-RUN] git -C ${HF_LOCAL_REPO} add models"
    echo "[DRY-RUN] git -C ${HF_LOCAL_REPO} status --short"
    echo "[DRY-RUN] git -C ${HF_LOCAL_REPO} commit -m \"${COMMIT_MESSAGE}\""
    if [[ "${PUSH}" == "1" ]]; then
      echo "[DRY-RUN] git -C ${HF_LOCAL_REPO} push"
    fi
    return 0
  fi

  git -C "${HF_LOCAL_REPO}" add models

  if [[ -z "$(git -C "${HF_LOCAL_REPO}" status --short)" ]]; then
    log "No changes to commit."
    return 0
  fi

  git -C "${HF_LOCAL_REPO}" status --short
  git -C "${HF_LOCAL_REPO}" commit -m "${COMMIT_MESSAGE}"

  if [[ "${PUSH}" == "1" ]]; then
    git -C "${HF_LOCAL_REPO}" push
  else
    log "Commit created locally. PUSH=0, so no remote push was performed."
  fi
}

main() {
  if [[ ! -d "${RESULTS_ROOT}" ]]; then
    echo "Results root not found: ${RESULTS_ROOT}" >&2
    exit 1
  fi

  log "Hugging Face repo: ${HF_REPO_ID}"
  log "Results root: ${RESULTS_ROOT}"
  log "Local clone: ${HF_LOCAL_REPO}"
  log "Model order: ${MODEL_LIST[*]}"
  log "DRY_RUN=${DRY_RUN} PUSH=${PUSH}"

  ensure_repo

  for model in "${MODEL_LIST[@]}"; do
    upload_model_group "${model}"
  done

  stage_commit_push
  log "Upload script finished."
}

main "$@"
