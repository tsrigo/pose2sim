# Batch Pose And Parallel Triangulation

This document explains the new throughput-oriented options added to Pose2Sim's `poseEstimation()` and `triangulation()` stages.

## What Changed

Pose estimation now has three layers of acceleration:

- `pose.batch_size`: batches frame detection and pose crops before ONNX inference.
- `pose.parallel_workers_pose`: keeps the existing per-video parallelism.
- `pose.det_frequency`: still controls how often the detector runs, but now interacts with batching.

Triangulation now has one new layer of acceleration:

- `triangulation.parallel_triangulation`: parallelizes per-frame triangulation with `ProcessPoolExecutor`.

## Config Knobs

Add or update these keys in `Config.toml`:

```toml
[pose]
backend = 'onnxruntime'
device = 'cuda'
display_detection = false
parallel_workers_pose = 'auto'
det_frequency = 1
batch_size = 8

[triangulation]
parallel_triangulation = 'auto'
```

Recommended starting point on GPU:

- `batch_size = 8`
- `parallel_workers_pose = 'auto'`
- `det_frequency = 1` when you want the largest end-to-end batching gain
- `parallel_triangulation = 'auto'`

## How The Pose Path Works

When `batch_size > 1`, Pose2Sim switches from the legacy per-frame loop to a batched video path.

At a high level:

1. Read up to `batch_size` frames.
2. Batch detector inputs when the detector ONNX model supports batched input.
3. Collect all person crops from the batch and run one batched pose inference.
4. Keep tracking, NMS, JSON export, and optional rendering in frame order.

This preserves correctness where ordering matters while moving the heavy ONNX work into larger inference calls.

## When Speedup Is Largest

Best case:

- `backend = 'onnxruntime'`
- `device = 'cuda'`
- `display_detection = false`
- `det_frequency = 1`
- multi-camera or multi-video workloads
- multiple visible people per frame, because crop batching grows with person count

Why `det_frequency = 1` matters:

- With `det_frequency = 1`, detector and pose inference both batch cleanly across frames.
- With `det_frequency > 1`, Pose2Sim preserves the legacy bbox propagation semantics between detector frames. This still allows batched pose crops, but detector-side gains are smaller because bbox state remains sequential.

## Fallback Behavior

The implementation is intentionally conservative.

- Non-`onnxruntime` backends fall back to sequential inference for the affected stage.
- Fixed-batch ONNX inputs such as `[1, 3, H, W]` fall back to sequential inference for that stage.
- One-stage pipelines without a detector model, such as RTMO-style paths, fall back to the legacy sequential video loop.
- Batched ONNX failures that look like memory pressure trigger chunk-size backoff and retry with a smaller batch.

This means you can set `batch_size > 1` globally and still get correct results even when a specific model cannot batch.

## Detector And Pose Models Can Behave Differently

Detector batching and pose batching are checked independently.

Typical outcomes:

- detector dynamic batch + pose dynamic batch: both stages batch
- detector fixed batch + pose dynamic batch: detector stays sequential, pose still batches crops
- detector dynamic batch + pose fixed batch: detector batches, pose falls back
- both fixed batch: behavior stays correct but effectively matches the old loop

## Triangulation Parallelism

`parallel_triangulation` parallelizes pure per-frame triangulation work.

- `'auto'`: use up to `os.cpu_count()` workers, capped by frame count
- integer: use that many workers
- `false`: disable process parallelism

Important detail:

- multi-person cross-frame re-ordering still runs sequentially after per-frame workers return

This keeps the frame-to-frame identity logic stable while parallelizing the CPU-heavy per-frame reconstruction.

## Suggested Rollout

For first deployment:

1. Turn off live display.
2. Set `batch_size = 8`.
3. Start with `det_frequency = 1` to measure the full batching path.
4. Enable `parallel_triangulation = 'auto'`.
5. Compare runtime and output on a short clip before scaling up.

If VRAM is tight:

- drop `batch_size` from `8` to `4`
- keep `parallel_workers_pose` modest
- increase only after confirming stable runs

## Output Compatibility

The new path keeps the same output contract:

- pose results are still written as OpenPose-style JSON
- tracking order is still applied per frame
- triangulation still produces the same TRC output format

The change is about throughput, not a new file format.
