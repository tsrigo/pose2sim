# PnVision action02 standing-stability rerun

## Reproduction

- Input videos: `data/PnVision/20260507/rec_20260507_134131_lhl_action02/cam01.avi` ... `cam04.avi`.
- Real entrypoint: `python -m Pose2Sim.Utilities.avi_to_trc --trial-dir data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02 --batch-size 8 --overwrite-pose`.
- `avi_to_trc` log confirms `2D temporal smoothing enabled for avi_to_trc`.
- Four cameras each produced 368 OpenPose JSON files; no frames were dropped and no detection misses were logged.
- The bad rerun allowed two-camera triangulation. For `Neck`, the best reprojection solution alternated between two camera pairs, which produced two valid-looking 3D branches and visible jumping.
- The final rerun keeps the same 2D JSON path but requires at least 3 cameras for triangulation and uses a 20 px reprojection threshold. This prevents the `Neck` two-camera branch switching instead of merely smoothing it after the fact.
- The final `filter_all` output still uses standard 3D outlier rejection and Butterworth filtering at 3 Hz, plus the pelvis rigid guard and short pairwise-spike repair.

## Outputs

- Rerun triangulated TRC: `data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02/pose-3d/rec_20260507_134131_lhl_action02_0-367.trc`.
- Final standing-stability TRC: `data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02/pose-3d/rec_20260507_134131_lhl_action02_0-368_filt_butterworth.trc`.
- Bad two-camera-allowed unfiltered TRC kept for comparison: `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/unfiltered_min2_before_min3_triangulation.trc`.
- Bad two-camera-allowed filtered TRC kept for comparison: `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/final_before_min3_triangulation.trc`.

## Metrics

Marker step stability:

| marker | bad min-2 unfiltered max step (mm) | bad min-2 filtered max step (mm) | final min-3 filtered max step (mm) | final max second diff (mm) |
|---|---:|---:|---:|---:|
| Neck | 79.99 | 33.11 | 32.40 | 5.04 |
| Hip | 42.29 | 38.00 | 38.00 | 13.41 |
| RHip | 114.11 | 38.10 | 43.31 | 24.12 |
| LHip | 86.53 | 38.27 | 38.28 | 20.92 |

Pelvis shape stability:

| metric | existing baseline TRC | bad min-2 triangulated | bad min-2 filtered | final min-3 filtered |
|---|---:|---:|---:|---:|
| pelvis pair std mean (mm) | 14.26 | 7.66 | 3.00 | 1.15 |
| pelvis pair maxdev mean (mm) | 94.07 | 56.01 | 27.53 | 11.06 |
| 253->254 pair change mean (mm) | 13.94 | 72.28 | 3.18 | 0.42 |

The final output removes the `Neck` two-camera branch switching at the triangulation stage. The previous filtered output made the jump smaller, but it still smoothed over the wrong branch changes. The final min-3 triangulation avoids those branch changes before filtering.

![Marker step stability](../figures/pnvision_action02_rtmpose_smoothing_20260519/target_marker_step_stability.png)

![Rerun pelvis distances](../figures/pnvision_action02_rtmpose_smoothing_20260519/target_pairwise_distances_rerun.png)

## Raw Evidence

- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_standing_stability_summary.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_marker_step_metrics.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_marker_step_timeseries.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_pairwise_distance_metrics.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_pairwise_distance_timeseries.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_253_254_transition.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/target_filter_rigid_stats.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/pose_json_counts.csv`
- `data/pnvision_action02_rtmpose_smoothing_20260519/evidence/run_Config.toml`
- `data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02/logs.txt`
