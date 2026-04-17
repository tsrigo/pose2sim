# Batch Pose And Parallel Triangulation Context

This folder records local implementation evidence for the batch pose inference and parallel triangulation milestone.

It is intentionally local and PR-oriented.

## Scope Included In The PR

- `Pose2Sim/poseEstimation.py`
- `Pose2Sim/triangulation.py`
- `Pose2Sim/Demo_SinglePerson/Config.toml`
- `Pose2Sim/Demo_MultiPerson/Config.toml`
- `Pose2Sim/Demo_Batch/Config.toml`
- `Pose2Sim/Demo_Batch/Trial_1/Config.toml`
- `Pose2Sim/Demo_Batch/Trial_2/Config.toml`
- `docs/README.md`
- `docs/batch-pose-and-parallel-triangulation.md`
- `context/batch-pose-parallel-triangulation-2026-04-17/*`

## Local Changes Explicitly Excluded From The PR

These files existed in the working tree but are not part of this milestone scope:

- `pyproject.toml`
- `Pose2Sim/Utilities/avi_to_trc.py`
- `data/`

## Environment

- date: `2026-04-17`
- repo HEAD before PR branching: `614804c`
- Python: `3.13.5`
- onnxruntime: `1.24.4`
- providers available locally: `['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- OpenCV: `4.13.0`

## Static Verification

Commands run:

```bash
python -m py_compile Pose2Sim/poseEstimation.py Pose2Sim/triangulation.py
python - <<'PY'
import Pose2Sim.poseEstimation
import Pose2Sim.triangulation
print("imports_ok")
PY
```

Observed result:

- compile succeeded
- module import succeeded

## ONNX Batch Capability Probe

Command summary:

- instantiated `BodyWithFeet` lightweight tracker on `onnxruntime` CPU
- inspected `session.get_inputs()[0].shape` for detector and pose models

Observed shapes:

- detector input shape: `[1, 3, 416, 416]`
- pose input shape: `['batch', 3, 256, 192]`

Interpretation:

- the lightweight detector model used in this local run does not support dynamic batching
- the lightweight pose model does support dynamic batching
- expected runtime behavior is therefore detector fallback + batched pose crops

## Local Runtime Verification

Verification project:

- copied `Pose2Sim/Demo_SinglePerson` into a temporary directory
- limited the run to frames `0..3`
- forced `backend = 'onnxruntime'`, `device = 'cpu'`, `display_detection = false`, `save_video = 'none'`

### Pose Estimation Run

Config overrides:

- `mode = 'lightweight'`
- `det_frequency = 1`
- `batch_size = 2`
- `parallel_workers_pose = 1`
- `overwrite_pose = true`

Observed result:

- `Pose2Sim.poseEstimation(config)` completed successfully
- generated `4` JSON files for each of `cam01`, `cam02`, `cam03`, and `cam04`
- wall time on this local CPU-only verification run: `4.236 s`

Key log evidence:

```text
Inference run on every single frame.
GPU pose batching requested with batch_size=2.
Detection ONNX model has fixed batch=1, falling back to sequential inference for this stage.
```

Meaning:

- the new batched path was entered
- the detector fallback guard was exercised on a real model
- pose estimation still completed successfully and produced output

### Triangulation Run

Config overrides:

- `parallel_triangulation = 2`
- `min_chunk_size = 1`
- `make_c3d = false`

Extra setup:

- the temporary demo was wrapped in a session-like parent directory expected by the existing calibration lookup code

Observed result:

- `Pose2Sim.triangulation(config)` completed successfully
- log showed `Triangulating frames in parallel with 2 worker processes.`
- generated TRC output: `Demo_SinglePerson_0-3.trc`
- wall time on this local CPU-only verification run: `1.689 s`

Key log evidence:

```text
Triangulating frames in parallel with 2 worker processes.
3D coordinates are stored at .../pose-3d/Demo_SinglePerson_0-3.trc.
```

## Notes And Limits

- This context proves the new batch and parallel code paths execute successfully in a real local run.
- It does not claim a GPU speedup number yet.
- The user-provided validation target based on `data/20260402/recordings/PNV-ITM-001` has not been run as part of this PR.
- The local CPU-only demo run exposed one useful compatibility fact: real speedup depends on the detector ONNX model having a dynamic batch dimension. The implementation now handles fixed-batch detectors safely instead of failing.
