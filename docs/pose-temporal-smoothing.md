# 2D temporal smoothing synthetic validation

## Setup

- Synthetic 240-frame single-person sequence at 30 fps.
- Person is nearly static, with small slow sway, Gaussian detector jitter, and occasional pelvis/foot spikes.
- The same `Pose2DTemporalSmoother` used by `Pose2Sim/poseEstimation.py` is applied before JSON writing.

## Result

| metric | raw | smoothed |
|---|---:|---:|
| frame step RMS (px) | 4.69 | 0.84 |
| second diff RMS (px) | 8.12 | 1.23 |
| frame step p95 (px) | 7.92 | 1.41 |

![Synthetic 2D smoothing](../figures/pose_temporal_smoothing/synthetic_pose2d_smoothing.png)

## Pipeline Smoke Test

- A synthetic MP4 was passed through the real `process_video` JSON-writing path with a deterministic dummy RTMPose tracker.
- Hip and foot keypoint frame-step RMS changed from 3.23 px to 0.56 px with temporal smoothing enabled.
- The target PnVision trial in this workspace has TRC output only; no source video or 2D RTMPose JSON is available here, so this 2D-stage validation uses synthetic input.

## Raw Evidence

- `data/pose_temporal_smoothing/synthetic_pose2d_metrics.csv`
- `data/pose_temporal_smoothing/synthetic_pose2d_timeseries.csv`
- `data/pose_temporal_smoothing/process_video_smoke_metrics.csv`
- `data/pose_temporal_smoothing/process_video_smoke/raw/pose/cam0_json/`
- `data/pose_temporal_smoothing/process_video_smoke/smoothed/pose/cam0_json/`
