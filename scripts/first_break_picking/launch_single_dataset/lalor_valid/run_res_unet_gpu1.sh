#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/first_break_picking/train_pick_res_unet.py"
CONFIG_DIR="${REPO_ROOT}/configs/first_break_picking/single_dataset/lalor_valid"

for seed in 42 43 44; do
  CONFIG="${CONFIG_DIR}/pick_res_unet_lalor_valid_seed${seed}.yaml"
  echo "Running res_unet on lalor_valid, seed ${seed}, GPU 1"
  CUDA_VISIBLE_DEVICES=1 python "${PY_SCRIPT}" --config "${CONFIG}"
done
