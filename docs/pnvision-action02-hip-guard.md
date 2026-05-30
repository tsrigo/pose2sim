# PnVision action02 hip guard report

## Reproduced issue

- Target: `data/PnVision/20260507/rec_20260507_134131_lhl_action02/pose-3d/rec_20260507_134131_lhl_action02.trc`.
- The stored filtered TRC does not show a large final-30-frame pelvis-distance spike, but the same trial has a clear pelvis-shape failure around frames 142-151.
- Baseline target pelvis pairwise-distance std: 14.26 mm.
- Baseline target mean max distance deviation: 94.07 mm.

## Root cause

The pelvis markers are triangulated and filtered as independent points. A short-lived 2D/keypoint or camera-selection error can keep an acceptable per-marker reprojection error while still collapsing the local `Hip/RHip/LHip` shape.

## Code changed

- `Pose2Sim/filtering.py` now guards 3D rigid marker stabilization with:
  - per-marker correction cap: 50 mm by default,
  - pairwise ratio protection for already-stable frames: 15% by default,
  - per-frame correction-vector step cap: 15 mm by default,
  - rejection of rigid corrections that move an outlier frame farther away from the learned template.
- The behavior remains opt-in through `filtering.rigid_marker_groups`; projects without this config keep the previous output path.

## Before / after

Target trial through real `filter_all`:

| metric | baseline | guarded |
|---|---:|---:|
| pairwise distance std mean (mm) | 14.26 | 7.81 |
| pairwise max deviation mean (mm) | 94.07 | 63.63 |
| final-30-frame distance std mean (mm) | 0.75 | 0.22 |
| pelvis jitter mean (mm/frame^2) | 6.09 | 6.04 |

All 27 PnVision TRCs in-memory:

| metric | baseline | guarded |
|---|---:|---:|
| pairwise distance std mean (mm) | 19.43 | 12.39 |
| pairwise max deviation mean (mm) | 94.34 | 71.71 |
| final-30-frame distance std mean (mm) | 7.98 | 6.27 |
| pelvis jitter mean (mm/frame^2) | 5.99 | 5.11 |

Target guard stats: accepted 368/368, applied 351/368, correction-limited frames 28, step-limited frames 26, rejected non-improving pairwise corrections 21.

## Figures

![Target pairwise distances](../figures/pnvision_action02_hip_guard/target_pairwise_distances.png)

![All trials pairwise std](../figures/pnvision_action02_hip_guard/all_trials_pairwise_std.png)

## Raw evidence

- Real `filter_all` output: `data/pnvision_action02_hip_guard/real_entrypoint_run/pose-3d/rec_20260507_134131_lhl_0-368_filt_butterworth.trc`.
- Target metrics: `data/pnvision_action02_hip_guard/target_before_after_metrics.csv`.
- Target pairwise time series: `data/pnvision_action02_hip_guard/target_pairwise_distance_timeseries.csv`.
- Target worst frames: `data/pnvision_action02_hip_guard/target_worst_pairwise_frames.csv`.
- All-trial metrics: `data/pnvision_action02_hip_guard/pnvision_all_trials_guarded_metrics.csv`.
- All-trial guard stats: `data/pnvision_action02_hip_guard/pnvision_all_trials_guarded_stats.csv`.
