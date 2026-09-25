#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/first_break_picking/train_pick_unet.py"
CONFIG_DIR="${REPO_ROOT}/configs/first_break_picking/single_dataset/dongbei"

for seed in 42 43 44; do
  CONFIG="${CONFIG_DIR}/pick_unet_dongbei_seed${seed}.yaml"
  echo "Running unet on dongbei, seed ${seed}, GPU 0"
  CUDA_VISIBLE_DEVICES=0 python "${PY_SCRIPT}" --config "${CONFIG}"
done
