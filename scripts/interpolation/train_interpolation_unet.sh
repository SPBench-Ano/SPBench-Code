#!/usr/bin/env bash
# 循环多次运行 scripts/train_interpolation_unet.py（torchrun + DDP）；每次只改 seed，
# experiment.name 变为 ``<yaml 中的 name>_seed<seed>``，输出目录互不覆盖。
#
# 直接在下面「配置区」改数值即可，无需命令行传参。

set -euo pipefail

# ---------- 配置区（按需修改）----------
CUDA_VISIBLE_DEVICES="6,7" # 物理 GPU，逗号分隔
NPROC_PER_NODE=2           # 须与可见 GPU 个数一致
N=5                        # 循环次数
START_SEED=42              # 第 1 次 seed，之后为 START_SEED+1, ...
MASK_MODE="uniform"        # uniform | random | continuous
MASK_RATIO="0.5"           # 缺失率，范围 (0, 1)
MASTER_PORT="auto"         # 固定端口号，或 "auto" 每次自动找空闲端口
TORCHRUN_EXTRA=""          # 可选：追加给 torchrun，例如 "--standalone"
# ------------------------------------

_pick_port() {
  python -c "import socket; s=socket.socket(); s.bind(('',0)); print(s.getsockname()[1]); s.close()"
}

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/interpolation/interpolation_unet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/interpolation/train_interpolation_unet.py"

export CUDA_VISIBLE_DEVICES

if [[ ! -f "${BASE_CONFIG}" ]]; then
  echo "Config not found: ${BASE_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${PY_SCRIPT}" ]]; then
  echo "Script not found: ${PY_SCRIPT}" >&2
  exit 1
fi

NAME_BASE="$(grep -m1 -E '^[[:space:]]*name:[[:space:]]*' "${BASE_CONFIG}" | sed -E 's/^[[:space:]]*name:[[:space:]]*//' | sed -E 's/[[:space:]]+#.*$//;s/[[:space:]]*$//')"
if [[ -z "${NAME_BASE}" ]]; then
  echo "Could not parse experiment.name from ${BASE_CONFIG}" >&2
  exit 1
fi

tmpcfg="$(mktemp)"
cleanup() { rm -f "${tmpcfg}"; }
trap cleanup EXIT

for ((i = 0; i < N; i++)); do
  seed=$((START_SEED + i))
  run_name="${NAME_BASE}_seed${seed}"
  sed -E \
    -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
    -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
    "${BASE_CONFIG}" >"${tmpcfg}"
  echo "[$(date -Iseconds)] (${i}+1)/${N} name=${run_name} seed=${seed}"
  cd "${REPO_ROOT}"
  if [[ "${MASTER_PORT}" == "auto" ]]; then
    master_port=$(_pick_port)
  else
    master_port="${MASTER_PORT}"
  fi
  # shellcheck disable=SC2086
  torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" \
    --master_port="${master_port}" \
    "${PY_SCRIPT}" --config "${tmpcfg}" \
    --mask-mode "${MASK_MODE}" --mask-ratio "${MASK_RATIO}"
done

echo "[$(date -Iseconds)] Done ${N} runs (seed ${START_SEED}..$((START_SEED + N - 1)))."
