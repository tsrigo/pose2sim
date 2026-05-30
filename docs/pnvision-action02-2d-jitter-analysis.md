# PnVision action02 2D RTMPose 源头抖动分析

## 范围

试次：`rec_20260507_134131_lhl_action02`。

本次分析使用真实源视频和仓库实际跑出的 RTMPose JSON：

- 源头 raw 2D：`data/pnvision_action02_neck_regression_20260519/no2d_smoothing/rec_20260507_134131_lhl_action02`（`temporal_smoothing = false`）
- 2D 平滑对照：`data/pnvision_action02_rtmpose_smoothing_20260519/rec_20260507_134131_lhl_action02`（`temporal_smoothing = true`）
- `det_frequency = 1` 小实验：`data/pnvision_action02_2d_jitter_detfreq1/rec_20260507_134131_lhl_action02`

主分析段是肉眼静止的开头 0-60 帧。contact sheet 显示这段人基本站立，后面才开始走动/转身。

## 汇报用短结论

医生看到的 2D 点抖动不是 TRC 或三维重建才产生的，raw RTMPose JSON 里已经存在。开头站立段里，抖动最大的是脚尖/脚跟等末端点，髋部和颈部也有可见跳动；不少大跳动的置信度仍然很高，所以单靠 likelihood 阈值筛不掉。

根因更接近 top-down 2D 姿态估计本身的逐帧不稳定：每帧都从图像 crop 里独立预测热图峰值，衣服边缘、鞋/地面接触、遮挡和视角会让同一个静止点在像素上来回选峰。2D 时序平滑能显著改善展示效果，但它是显示/质量控制层面的稳定化，不等于模型本身已经可靠。

## 关键证据

raw 2D 在 0-60 帧中逐帧位移 p95 最大的点：

| camera   | keypoint   |   step_p50_px |   step_p95_px |   step_max_px |   median_score_min |   large_step_over_5px |
|:---------|:-----------|--------------:|--------------:|--------------:|-------------------:|----------------------:|
| cam02    | LSmallToe  |          5.38 |         15.74 |         17.24 |               0.70 |                  0.57 |
| cam02    | LBigToe    |          4.80 |         12.09 |         16.88 |               0.75 |                  0.50 |
| cam04    | LBigToe    |          3.78 |          8.46 |          8.79 |               0.87 |                  0.37 |
| cam04    | LSmallToe  |          3.78 |          7.61 |          8.45 |               0.93 |                  0.32 |
| cam04    | LHeel      |          3.77 |          7.61 |         11.01 |               0.88 |                  0.27 |
| cam04    | LAnkle     |          3.78 |          7.56 |          8.44 |               0.89 |                  0.20 |
| cam04    | RSmallToe  |          3.77 |          7.56 |         11.66 |               0.74 |                  0.25 |
| cam02    | RAnkle     |          3.70 |          5.47 |          7.76 |               0.86 |                  0.12 |
| cam02    | RSmallToe  |          3.81 |          5.47 |          7.37 |               0.90 |                  0.20 |
| cam04    | RBigToe    |          3.78 |          5.42 |          7.56 |               0.79 |                  0.20 |
| cam02    | RHeel      |          3.81 |          5.42 |          8.66 |               0.86 |                  0.18 |
| cam02    | RBigToe    |          1.43 |          5.40 |          5.56 |               0.92 |                  0.12 |
| cam02    | LHeel      |          3.81 |          5.40 |          7.73 |               0.83 |                  0.10 |
| cam02    | LHip       |          0.00 |          5.37 |          5.59 |               0.67 |                  0.07 |

Neck 在各相机 raw 2D 里的抖动：

| camera   |   step_p50_px |   step_p95_px |   step_max_px |   median_score_min |   large_step_over_5px |
|:---------|--------------:|--------------:|--------------:|-------------------:|----------------------:|
| cam02    |          0.00 |          3.87 |          5.12 |               0.93 |                  0.02 |
| cam04    |          0.00 |          3.79 |          7.75 |               0.99 |                  0.03 |
| cam03    |          0.00 |          1.53 |          2.03 |               1.00 |                  0.00 |
| cam01    |          0.00 |          1.43 |          2.02 |               0.99 |                  0.00 |

raw 与 2D 平滑后的关注点汇总：

| keypoint   |   mean_step_p95_px_raw |   mean_step_p95_px_smooth |   step_p95_reduction_pct |   mean_radial_p95_px_raw |   mean_radial_p95_px_smooth |   radial_p95_reduction_pct |
|:-----------|-----------------------:|--------------------------:|-------------------------:|-------------------------:|----------------------------:|---------------------------:|
| LSmallToe  |                   6.79 |                      1.55 |                    77.21 |                     5.82 |                        3.64 |                      37.42 |
| LBigToe    |                   6.01 |                      1.57 |                    73.86 |                     5.86 |                        4.39 |                      25.08 |
| LAnkle     |                   4.51 |                      0.80 |                    82.34 |                     3.35 |                        1.84 |                      44.99 |
| LHip       |                   4.26 |                      1.18 |                    72.34 |                     4.77 |                        4.36 |                       8.52 |
| RSmallToe  |                   4.21 |                      0.81 |                    80.66 |                     3.63 |                        2.22 |                      38.95 |
| LHeel      |                   4.12 |                      0.82 |                    80.02 |                     4.45 |                        2.40 |                      46.08 |
| RHeel      |                   3.75 |                      0.72 |                    80.92 |                     4.04 |                        2.33 |                      42.40 |
| RHip       |                   3.69 |                      0.96 |                    74.04 |                     4.42 |                        3.66 |                      17.10 |
| RAnkle     |                   3.63 |                      0.78 |                    78.48 |                     3.44 |                        2.85 |                      16.99 |
| RBigToe    |                   3.58 |                      0.79 |                    78.03 |                     3.10 |                        2.20 |                      28.87 |
| Hip        |                   2.89 |                      0.59 |                    79.58 |                     3.27 |                        1.92 |                      41.30 |
| Neck       |                   2.65 |                      0.58 |                    78.22 |                     3.27 |                        2.14 |                      34.48 |

`det_frequency` 对照：补充对照把同一段 0-60 帧重跑为 `det_frequency = 1`，12/12 个关注点的平均 p95 跳动变大。因此 4 帧刷新不是主因；每帧重检会带来更多 bbox/crop 变化，反而可能放大 raw 2D 抖动。

| keypoint   |   mean_step_p95_px_det4 |   mean_step_p95_px_det1 |   p95_change_pct |   mean_radial_p95_px_det4 |   mean_radial_p95_px_det1 |
|:-----------|------------------------:|------------------------:|-----------------:|--------------------------:|--------------------------:|
| LBigToe    |                    6.01 |                    7.94 |            32.14 |                      5.86 |                      6.54 |
| RAnkle     |                    3.63 |                    4.66 |            28.57 |                      3.44 |                      3.87 |
| RHeel      |                    3.75 |                    4.67 |            24.48 |                      4.04 |                      3.95 |
| RBigToe    |                    3.58 |                    4.19 |            17.18 |                      3.10 |                      3.32 |
| LHeel      |                    4.12 |                    4.78 |            15.99 |                      4.45 |                      4.50 |
| RSmallToe  |                    4.21 |                    4.82 |            14.36 |                      3.63 |                      4.02 |
| RHip       |                    3.69 |                    4.10 |            11.02 |                      4.42 |                      5.20 |
| LSmallToe  |                    6.79 |                    7.44 |             9.55 |                      5.82 |                      5.86 |
| Hip        |                    2.89 |                    3.11 |             7.73 |                      3.27 |                      3.13 |
| Neck       |                    2.65 |                    2.78 |             4.89 |                      3.27 |                      3.15 |
| LAnkle     |                    4.51 |                    4.54 |             0.63 |                      3.35 |                      3.49 |
| LHip       |                    4.26 |                    4.28 |             0.38 |                      4.77 |                      5.54 |

检测刷新节奏检查：当前 `det_frequency = 4` 时，`frame % 4 == 0` 是检测框刷新边界。刷新边界的平均 p95 位移是非刷新边界的 1.01x，整体没有明显的 4 帧周期峰值；个别点位会受 bbox/crop 影响，但不是全局主因。

置信度检查：raw 关注点在 0-60 帧内，8.2% 的有效逐帧位移超过 5 px，其中 7.6% 同时满足位移 >5 px 且两端最小 likelihood >=0.7。这说明“大跳动”并不总是低置信度点，医生看到的抖动不能只靠置信度解释。

## 图和视频证据

![Static focus step p95](../figures/pnvision_action02_2d_jitter/focus_static_step_p95_by_keypoint.png)

![Detector refresh cadence](../figures/pnvision_action02_2d_jitter/det_refresh_modulo_static_step.png)

![det_frequency comparison](../figures/pnvision_action02_2d_jitter/detfreq1_vs_detfreq4_static_step.png)

![Neck trace](../figures/pnvision_action02_2d_jitter/neck_trace_raw_vs_smoothed.png)

![Static scatter examples](../figures/pnvision_action02_2d_jitter/static_scatter_examples.png)

![Confidence vs step](../figures/pnvision_action02_2d_jitter/confidence_vs_step_static.png)

代表性 overlay 视频：`figures/pnvision_action02_2d_jitter/cam02_static_raw_vs_smoothed_overlay.mp4`。左边是 raw RTMPose 关注点，右边是 2D 平滑后关注点，相机 `cam02`。

用于确认静止段的 contact sheet：

- `figures/pnvision_action02_2d_jitter/cam01_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam02_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam03_contact_sheet.jpg`
- `figures/pnvision_action02_2d_jitter/cam04_contact_sheet.jpg`

## 根因判断

1. 这不是 TRC-only artifact。raw JSON 在站立段已经有逐帧 2D 位移，并且能直接画回源视频。
2. `det_frequency = 4` 不是唯一主因。重跑 `det_frequency = 1` 后，多数关注点更抖，说明频繁重检本身也会引入 crop 变化。
3. RTMPose raw 输出没有时间一致性约束。Neck、pelvis、feet 都是逐帧热图选峰，局部视觉证据相似时会在相邻像素峰之间跳。
4. likelihood 不是充分的稳定性指标。一部分明显跳动仍是高置信度，所以低置信度过滤只能覆盖一部分问题。

## 建议的下一步

- 面向医生的 2D 录像默认使用“平滑 + 质量标记”的 overlay；raw overlay 保留为 debug 证据，不直接当稳定测量展示。
- 每个 trial 自动出 2D 抖动质控报告：p95 逐帧位移、静止段散布、高置信大跳动比例、多相机一致性。
- 在 pose 估计前做 bbox/crop 稳定化实验：平滑目标框中心/尺度，或使用跟踪框，再和当前 raw JSON 做同段对照。
- 2D 平滑保持配置化/自适应，避免快速动作被过度平滑。
- 不再用 likelihood 单独说明点位可靠；需要结合时间一致性和多相机几何一致性。

## 限制

当前 JSON 不保存 bbox/crop 参数，所以 bbox 贡献是通过 `det_frequency` 对照和帧序节奏间接判断的。要精确证明 bbox 抖动幅度，需要下一步在 RTMPose 导出阶段记录每帧 bbox center/scale。

## 原始证据文件

- `data/pnvision_action02_2d_jitter_analysis/pose2d_points.csv`
- `data/pnvision_action02_2d_jitter_analysis/pose2d_frame_steps.csv`
- `data/pnvision_action02_2d_jitter_analysis/static_step_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/static_spread_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/focus_keypoint_summary.csv`
- `data/pnvision_action02_2d_jitter_analysis/detector_modulo_metrics.csv`
- `data/pnvision_action02_2d_jitter_analysis/detfreq1_comparison.csv`
