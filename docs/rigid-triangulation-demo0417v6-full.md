# demo0417v6 完整真实数据刚体组三角化验证报告

## 复现内容

- 数据来自用户提供的 `data/demo0417v6.zip`，不是合成数据。
- 视频帧数：`cam01=467`、`cam02=391`、`cam03=467`、`cam04=467`；为了四路相机同步对比，完整验证使用最短相机范围 `frame_range = [0, 391]`。
- 以 30 fps 计算，本报告覆盖约 13.03 秒、391 帧纯站立视频。
- 使用真实 `Pose2Sim.poseEstimation()` 生成的 2D JSON，再分别运行 baseline `Pose2Sim.triangulation()` 与启用刚体组后的 `Pose2Sim.triangulation()`。
- baseline TRC 和 rigid TRC 使用同一批 2D JSON、同一标定文件、同一帧范围。

## 根因与改动

- 原三角化逐个 marker 独立选择相机并独立估计 3D 点，`Hip/LHip/RHip` 与头面部 marker 的相对距离会逐帧变化。
- 修正后的 `triangulation.rigid_marker_groups` 仍先按原流程得到 baseline，再从稳定帧估计组内模板，逐帧联合最小化该组所有有效 2D 重投影误差。
- 输出不再 100% 替换成刚体模板，而是用 `rigid_group_blend` 混回 baseline，并用 `rigid_group_max_correction_m` 与 `rigid_group_max_pairwise_change_ratio` 防止髋部比例被压缩。
- rigid 拟合默认先用全部有效相机，失败时最多排除 1 个相机，避免在差帧上枚举大量相机组合。
- 这次只启用 `['Hip', 'LHip', 'RHip']` 和 `['Nose', 'LEye', 'REye', 'LEar', 'REar']`，没有对肩膀和脚部 marker 加约束。

## Before / After

- 髋部三点平均组内距离标准差：7.81 -> 2.18 mm。
- 髋部 pairwise 距离相对 baseline 的最大帧级变化：p95 <= 14.0%，max <= 15.0%。
- 头面部五点平均组内距离标准差：4.29 -> 1.47 mm。
- `Hip`: 7.60 -> 7.03 mm/frame² (-7.5%).
- `RHip`: 58.49 -> 23.09 mm/frame² (-60.5%).
- `LHip`: 41.58 -> 15.68 mm/frame² (-62.3%).

刚体输出相对 baseline 的空间改变量：
- `Hip`: median 1.74 mm, p95 14.07 mm.
- `RHip`: median 2.60 mm, p95 45.12 mm.
- `LHip`: median 2.10 mm, p95 32.29 mm.
- `Nose`: median 1.97 mm, p95 7.13 mm.
- `REye`: median 1.93 mm, p95 5.55 mm.
- `LEye`: median 2.15 mm, p95 16.33 mm.
- `REar`: median 1.63 mm, p95 3.08 mm.
- `LEar`: median 1.39 mm, p95 3.09 mm.

刚体拟合接受情况：
- `Hip+LHip+RHip`: accepted 391/391 frames, applied 391/391 frames, mean joint reprojection error 13.3 px, mean correction 6.8 mm, max correction 50.0 mm, elapsed 0.72 s.
- `Nose+LEye+REye+LEar+REar`: accepted 391/391 frames, applied 391/391 frames, mean joint reprojection error 8.6 px, mean correction 2.7 mm, max correction 18.4 mm, elapsed 0.89 s.

肩膀和脚部 marker 的抖动数值在本次实验中保持不变，这是预期结果：本次范围没有对这些 marker 施加刚体约束。

## Figures

![Hip trajectories](../figures/rigid_triangulation_demo0417v6_full/fig01_hip_trajectories.png)

![Pairwise stability](../figures/rigid_triangulation_demo0417v6_full/fig02_pairwise_distance_stability.png)

![Marker jitter](../figures/rigid_triangulation_demo0417v6_full/fig03_marker_jitter.png)

![Hip distance time series](../figures/rigid_triangulation_demo0417v6_full/fig04_hip_pairwise_distance_timeseries.png)

![Face distance time series](../figures/rigid_triangulation_demo0417v6_full/fig05_face_pairwise_distance_timeseries.png)

## Raw Evidence

- `data/rigid_triangulation_demo0417v6_full/baseline/demo0417v6_0-390_baseline.trc`
- `data/rigid_triangulation_demo0417v6_full/strict_before/demo0417v6_0-390_strict_before.trc`
- `data/rigid_triangulation_demo0417v6_full/rigid/demo0417v6_0-390_rigid.trc`
- `data/rigid_triangulation_demo0417v6_full/rigid_guarded/demo0417v6_0-390_rigid_guarded.trc`
- `data/rigid_triangulation_demo0417v6_full/pairwise_distance_metrics.csv`
- `data/rigid_triangulation_demo0417v6_full/pairwise_distance_change_metrics.csv`
- `data/rigid_triangulation_demo0417v6_full/marker_jitter_metrics.csv`
- `data/rigid_triangulation_demo0417v6_full/marker_displacement_metrics.csv`
- `data/rigid_triangulation_demo0417v6_full/rigid_acceptance_metrics.csv`
- `data/rigid_triangulation_demo0417v6_full/baseline_run.log`
- `data/rigid_triangulation_demo0417v6_full/strict_before/rigid_strict_before.log`
- `data/rigid_triangulation_demo0417v6_full/rigid_run.log`
- `data/rigid_triangulation_demo0417v6_full/rigid_guarded_run.log`
- `data/rigid_triangulation_demo0417v6_full/rigid_guarded_parallel_smoke.log`

## Limits

- 本报告验证的是可同步使用的完整 391 帧范围。`cam01/cam03/cam04` 还有额外帧，但 `cam02` 只有 391 帧，因此没有把后续单相机缺帧区间纳入四相机三角化对比。
- 这次没有更新下游 OpenSim marker/IK 设置。
- 如果还要直接降低肩膀或脚部 marker 抖动，需要另行配置肩带/足部刚体或半刚体组，风险和运动学约束需要单独评估。
