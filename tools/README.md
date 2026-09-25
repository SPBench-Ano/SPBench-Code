# tools/

## Purpose

Generic utilities and helpers reused across the project. No training pipelines and no model definitions here.

## Available modules

- `array_io.py` — Unified volume loader that dispatches by file extension.
  - `load_volume(data_cfg)` — reads `.npy`, `.mat` (via `scipy.io.loadmat`), or `.sgy`/`.segy` (via `segy_read.read_regular_shots`) and returns a 3-D float32 array `(n_shots, n_traces, n_time)`.
  - `read_npy_volume(path)`, `read_mat_volume(path, key=None)`, `read_sgy_volume(path, traces_per_shot, time_downsample=1)` — low-level readers.

- `segy_read.py` — SEG-Y reader that groups traces into shot gathers with output shape `(n_shots, shot_length, time_length)`. Built on [segyio](https://github.com/equinor/segyio).
  - `read_regular_shots(path, traces_per_shot, time_downsample=1, ...)` — default, equal trace count per shot; `traces_per_shot` is supplied by the caller and `n_shots` is derived; supports anti-aliased time downsampling via `scipy.signal.decimate` (typical value `time_downsample=2`).
  - `read_irregular_shots_by_header(path, header_key="FieldRecord", time_downsample=1, ...)` — placeholder; per-shot splitting is done by reading trace headers (`FieldRecord` and optionally `SourceX`/`SourceY`) and using `np.unique(..., return_index=True)` to find shot boundaries.
  - `inspect_segy(path)` — quick metadata probe (trace count, samples, sample interval, unique FFID count, per-FFID min/max trace count); use it first to infer a safe `traces_per_shot`.

- `preprocessing.py` — per-shot primitives on `(n_shots, n_traces, n_time)`; pure numpy, reproducible via an injected `rng`.
  - `add_noise(shots, kind, snr_db, rng)` — SNR-controlled IID noise; `kind="gaussian"` or `"poisson"`. SNR: `SNR_dB = 10*log10(var_signal/var_noise)`.
  - `add_linear_noise(shots, dt, *, rng)` — **NOT IMPLEMENTED YET**. Placeholder for coherent linear-moveout events (ground-roll-like).
  - `add_hyperbolic_noise(shots, dt, *, rng)` — **NOT IMPLEMENTED YET**. Placeholder for coherent hyperbolic-moveout events (multiples-like).
  - `mask_traces(shots, mode, ratio, *, uniform_stride=None, rng)` — zero traces; modes `"uniform"` / `"random"` / `"continuous"`.
  - `spherical_divergence_correction(shots, dt, t0=0.0, power=2.0)` — multiplies each sample by `(t + t0)**power`.
  - `normalize(shots, mode, clip_percentile=None, per="shot", override_stats=None)` — `"minmax"` / `"max_abs"` / `"mean_std"` with optional percentile clip. `override_stats` applies caller-supplied scalars (`{"min","max"} | {"max_abs"} | {"mean","std"}`); when set, `per` is forced to `"global"`. `override_stats` may also contain `"clip_threshold"` to replay an earlier symmetric clip on a second volume (e.g. paired denoise target).

- `patching.py` — patchify / unpatchify on `(trace, time)`; accepts 2D or 3D input; output is `(P, h, w)` (`output_ndim=3`) or `(P, 1, h, w)` (`output_ndim=4`), `P = n_shots * n_per_shot`.
  - `patchify_uniform(data, patch_size, overlap=0.0, output_ndim=3)` — overlapping grid with stride `max(1, round(patch*(1-overlap)))`; the last patch per axis is tail-anchored. Returns `(patches, info)` consumed by `unpatchify_uniform`.
  - `patchify_random(data, patch_size, n_patches, output_ndim=3, rng=None)` — `n_patches` **per shot** of independently sampled starts; start positions recorded in `info`. No inverse.
  - `unpatchify_uniform(patches, info)` — inverse of `patchify_uniform`; overlaps are averaged (`sum/count`); accepts 3D or 4D patches; raises on `info` from `patchify_random`.

- `_array_utils.py` — internal helpers (`as_3d`, `restore`, `as_generator`, `RNGLike`) shared by `preprocessing.py` and `patching.py`. Not part of the public API.

## Planned contents

- `io.py` — File I/O helpers (path handling, safe read/write, compression).
- `segy2h5.py` — SEG-Y → HDF5 conversion script (raw data preprocessing).
- `coords.py` — Coordinate utilities for inline / xline / offset, normalization.
- `env.py` — Infrastructure helpers: random seed, device selection, DDP detection.
- `registry.py` — Optional lightweight registry for model / dataset / loss lookup.

## Constraints

- Do not put business pipelines (training loops, augmentation policies, model structures) here.
- Each utility must be a pure function or a lightweight class with no implicit global state.
- Before adding a new utility, check both `utils/` and this directory to avoid duplication.

## How to add a new tool

1. Create a new file `tools/<tool_name>.py` that exposes pure functions or a thin class.
2. Import it from `tools/__init__.py` only if it should be part of the public API.
3. If the helper is training-related (datasets, losses, metrics, etc.), put it under `utils/` instead.
4. Update `memory/updates.md` if the tool affects multiple modules or depends on a new library.
