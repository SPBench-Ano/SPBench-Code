#!/usr/bin/env bash
# Wrapper for upload_random_model_to_hf.py so routine uploads can be controlled
# by editing this shell file only.
set -euo pipefail

# ---------- User-tunable configuration ----------
export HF_NAMESPACE="SPBench"
# export HF_TOKEN="your_hf_token"

REPO_NAME="random-noise-attenuation"
RESULTS_DIR="/data/pcdisk0/data/results/random_noise"

# Change upload order or limit to selected models here.
MODEL_LIST=("unet")

# Optional display names for the Hugging Face model card tables.
# Format: "model_key=Display Name"
MODEL_DISPLAY_LIST=(
  "unet=UNet"
  "dncnn=DnCNN"
  "res_unet=ResUNet"
  "atten_unet=Attention UNet"
  "SCRN=SCRN"
)

# 1 = dry-run, 0 = real upload
DRY_RUN=1

# 1 = skip uploading README.md, 0 = upload/update model card too
NO_MODEL_CARD=0

# Default model card template
MODEL_CARD="scripts/random_noise_suppression/README_models.md"
# -----------------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/random_noise_suppression/upload_random_model_to_hf.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Python uploader not found: ${PY_SCRIPT}" >&2
  exit 1
fi

CMD=(
  python
  "${PY_SCRIPT}"
  --repo-name "${REPO_NAME}"
  --results-dir "${RESULTS_DIR}"
  --model-card "${MODEL_CARD}"
)

if (( ${#MODEL_LIST[@]} > 0 )); then
  CMD+=(--models "${MODEL_LIST[@]}")
fi

if (( ${#MODEL_DISPLAY_LIST[@]} > 0 )); then
  for item in "${MODEL_DISPLAY_LIST[@]}"; do
    CMD+=(--display-name "${item}")
  done
fi

if [[ "${DRY_RUN}" == "1" ]]; then
  CMD+=(--dry-run)
fi

if [[ "${NO_MODEL_CARD}" == "1" ]]; then
  CMD+=(--no-model-card)
fi

echo "Running uploader with:"
echo "  HF_NAMESPACE=${HF_NAMESPACE:-}"
echo "  REPO_NAME=${REPO_NAME}"
echo "  RESULTS_DIR=${RESULTS_DIR}"
echo "  MODEL_LIST=${MODEL_LIST[*]}"
echo "  MODEL_DISPLAY_LIST=${MODEL_DISPLAY_LIST[*]}"
echo "  DRY_RUN=${DRY_RUN}"
echo "  NO_MODEL_CARD=${NO_MODEL_CARD}"
echo
printf 'Command:'
for arg in "${CMD[@]}"; do
  printf ' %q' "${arg}"
done
printf '\n'

cd "${REPO_ROOT}"
"${CMD[@]}"
