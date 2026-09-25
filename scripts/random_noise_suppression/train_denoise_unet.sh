#!/usr/bin/env bash
# Run random-noise-suppression training over multiple noise kinds, SNR levels,
# and seeds. Each run rewrites ``seed``, ``experiment.name``, ``noise_kind``,
# and ``snr_db`` in a temporary config so outputs never collide.
set -euo pipefail

unset DISPLAY
export MPLBACKEND=Agg

# ---------- Configuration ----------
CUDA_VISIBLE_DEVICES="0,1" # Physical GPU ids, comma-separated.
NPROC_PER_NODE=2           # Should match the number of visible GPUs.
N=3                        # Number of runs.
START_SEED=42              # First seed; later runs use START_SEED+1, ...
NOISE_KIND_LIST=("gaussian" "poisson")
SNR_LIST=(-5 0 5)          # Synthetic noise SNR values (dB).
TORCHRUN_EXTRA=""          # Optional extra torchrun args, e.g. "--standalone".
# -----------------------------------

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"
BASE_CONFIG="${REPO_ROOT}/configs/random_noise_suppression/denoise_unet.yaml"
PY_SCRIPT="${REPO_ROOT}/scripts/random_noise_suppression/train_denoise_unet.py"

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

for noise_kind in "${NOISE_KIND_LIST[@]}"; do
  for snr in "${SNR_LIST[@]}"; do
    if (( snr < 0 )); then
      snr_tag="neg$((-snr))"
    else
      snr_tag="${snr}"
    fi

    for ((i = 0; i < N; i++)); do
      seed=$((START_SEED + i))
      run_name="${NAME_BASE}_${noise_kind}_snr${snr_tag}_seed${seed}"
      sed -E \
        -e 's/^([[:space:]]*seed:[[:space:]]*)[0-9]+$/\1'"${seed}"'/' \
        -e 's/^([[:space:]]*name:[[:space:]]*).*/\1'"${run_name}"'/' \
        -e 's/^([[:space:]]*noise_kind:[[:space:]]*).*/\1'"${noise_kind}"'/' \
        -e 's/^([[:space:]]*snr_db:[[:space:]]*).*/\1'"${snr}"'/' \
        "${BASE_CONFIG}" >"${tmpcfg}"
      echo "[$(date -Iseconds)] noise_kind=${noise_kind} snr_db=${snr} (${i}+1)/${N} name=${run_name} seed=${seed}"
      cd "${REPO_ROOT}"
      # shellcheck disable=SC2086
      torchrun ${TORCHRUN_EXTRA} --nproc_per_node="${NPROC_PER_NODE}" "${PY_SCRIPT}" --config "${tmpcfg}"
    done
  done
done

echo "[$(date -Iseconds)] Done ${#NOISE_KIND_LIST[@]} noise kinds x ${#SNR_LIST[@]} SNR levels x ${N} seeds."
