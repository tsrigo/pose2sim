# Batched AVI to 3D TRC

`avi_to_trc` is a standalone CLI for Pose2Sim-style trials. It reads
multi-camera `*.avi` videos, runs RTMLib 2D pose estimation, writes
OpenPose-style JSONs, and then reuses Pose2Sim triangulation to generate a
standard 3D `.trc`.

## What It Does

- Input: one trial directory with `videos/*.avi`
- 2D inference: frame-wise person detection plus batched RTMPose crop inference
- Intermediate outputs: `pose/<camera>_json/*.json`
- Final output: `pose-3d/*.trc`

This is useful when the bottleneck is RTMPose inference on one synchronized
trial and you want better utilization than the legacy per-frame pose path.

## Directory Contract

The command expects a Pose2Sim-compatible trial layout:

```text
Session/
  Config.toml            # optional if Trial_1/Config.toml exists
  calibration/
    *.toml               # required before triangulation
  Trial_1/
    Config.toml          # optional if Session/Config.toml exists
    videos/
      cam01.avi
      cam02.avi
      cam03.avi
      cam04.avi
```

Important:

- The trial must contain `videos/*.avi`.
- A calibration `.toml` must already exist under the session calibration
  directory.
- v1 is single-person only.
- v1 only supports top-down RTMLib models with a detector plus pose model.
  One-stage RTMO-style configs are rejected.

## Basic Usage

```bash
avi_to_trc --trial-dir /path/to/Session/Trial_1 --batch-size 16 --overwrite-pose
```

Common options:

- `--config /path/to/Config.toml`
- `--batch-size 16`
- `--det-frequency 4`
- `--backend onnxruntime`
- `--device cuda`
- `--overwrite-pose`

If `--config` is omitted, the CLI resolves config in this order:

1. `Trial_1/Config.toml` merged onto `Session/Config.toml`
2. `Trial_1/Config.toml`
3. `Session/Config.toml`

## End-to-End Example

If calibration already exists:

```bash
avi_to_trc \
  --trial-dir /data/my_session/Trial_1 \
  --batch-size 16 \
  --backend onnxruntime \
  --device cuda \
  --overwrite-pose
```

Outputs:

- `Trial_1/pose/cam01_json/*.json`
- `Trial_1/pose/cam02_json/*.json`
- `Trial_1/pose/cam03_json/*.json`
- `Trial_1/pose/cam04_json/*.json`
- `Trial_1/pose-3d/<trial_name>_<start>-<end>.trc`

## Performance Model

The batching happens at pose inference, not at detection:

- Detection stays frame-by-frame because the default YOLOX detector ONNX shape
  is fixed to batch size `1`.
- RTMPose crops are accumulated and sent through the pose model as a batch.

In practice this means:

- `batch_size=1` behaves like a sequential crop path.
- Larger batch sizes improve pose throughput when VRAM or CPU memory allows.
- `det_frequency` still controls how often the detector refreshes the person
  box.

## Output Semantics

The generated `.trc` is a standard 3D OpenSim-compatible TRC produced by the
existing Pose2Sim triangulation module. This command does not write a fake 2D
TRC or a monocular pseudo-3D TRC.

## Failure Modes

`No calibration TOML file found`

- Run the calibration conversion/generation step first so the session has a
  calibration `.toml`.

`At least two camera JSON streams are required`

- Triangulation needs at least two synchronized cameras.

`avi_to_trc only supports top-down RTMLib models`

- Switch to a detector + RTMPose configuration such as the built-in
  `Body_with_feet`, `Whole_body`, or `Body`.

`Existing pose-sync/pose-associated JSON outputs`

- Use `--overwrite-pose` to regenerate pose outputs for this trial, or clean
  the old downstream pose folders first.

## Practical Notes

- The CLI forces `output_format = 'openpose'` because the triangulation module
  consumes OpenPose-style JSON.
- If camera videos have different lengths, the command clips processing to the
  common valid frame range and logs a warning.
- Existing compatible pose JSONs are reused unless `--overwrite-pose` is set.
