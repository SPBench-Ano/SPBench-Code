#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/first_break_picking/train_pick_atten_unet.py"
CONFIG_DIR="${REPO_ROOT}/configs/first_break_picking/single_dataset/brunswick_valid"

for seed in 42 43 44; do
  CONFIG="${CONFIG_DIR}/pick_atten_unet_brunswick_valid_seed${seed}.yaml"
  echo "Running atten_unet on brunswick_valid, seed ${seed}, GPU 2"
  CUDA_VISIBLE_DEVICES=2 python "${PY_SCRIPT}" --config "${CONFIG}"
done
