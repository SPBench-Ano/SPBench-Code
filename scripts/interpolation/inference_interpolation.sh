#!/usr/bin/env bash
# Interpolation inference wrapper: run inference on a trained checkpoint.
# Modify CONFIG / CHECKPOINT paths in the configuration block below.

set -euo pipefail

# ---------- 配置区（按需修改）----------
CUDA_VISIBLE_DEVICES="6"
CONFIG="configs/interpolation/interpolation_unet.yaml"
# 以下变量为空时，Python 脚本自动从 config.inference 读取对应值
CHECKPOINT=""          # 空 -> 使用 config.inference.checkpoint
OUTPUT_DIR=""          # 空 -> 使用 config.inference.output_dir
N_VIZ_SHOTS=""         # 空 -> 使用 config.inference.n_viz_shots
SEED=""                # 空 -> 使用 config.inference.seed（或 experiment.seed）
DEVICE=""              # 空 -> 使用 config.inference.device（或 experiment.device）
# ------------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/inference_interpolation.py"

if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

args=(--config "${REPO_ROOT}/${CONFIG}")

if [[ -n "${CHECKPOINT}" ]]; then
  args+=(--checkpoint "${CHECKPOINT}")
fi
if [[ -n "${OUTPUT_DIR}" ]]; then
  args+=(--output-dir "${OUTPUT_DIR}")
fi
if [[ -n "${N_VIZ_SHOTS}" ]]; then
  args+=(--n-viz-shots "${N_VIZ_SHOTS}")
fi
if [[ -n "${SEED}" ]]; then
  args+=(--seed "${SEED}")
fi
if [[ -n "${DEVICE}" ]]; then
  args+=(--device "${DEVICE}")
fi

cd "${REPO_ROOT}"
python "${PY_SCRIPT}" "${args[@]}"

echo "[$(date -Iseconds)] Inference finished."
