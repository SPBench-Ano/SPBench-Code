#!/usr/bin/env bash
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"
PY_SCRIPT="${REPO_ROOT}/scripts/first_break_picking/train_pick_dncnn_seg.py"
CONFIG_DIR="${REPO_ROOT}/configs/first_break_picking/single_dataset/dongbei"

for seed in 42 43 44; do
  CONFIG="${CONFIG_DIR}/pick_dncnn_seg_dongbei_seed${seed}.yaml"
  echo "Running dncnn_seg on dongbei, seed ${seed}, GPU 3"
  CUDA_VISIBLE_DEVICES=3 python "${PY_SCRIPT}" --config "${CONFIG}"
done
