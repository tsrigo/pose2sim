# demo0417v6 刚体组三角化验证报告

> 注：这是早期 180 帧 strict rigid 验证记录。当前三角化 rigid 已改为默认近似刚体，并加入比例保护；完整更新结果见 `docs/rigid-triangulation-demo0417v6-full.md`。

## 复现内容

- 数据来自 `data/demo0417v6.zip`，不是合成数据。
- 为了快速闭环，复现实验使用真实视频的 `frame_range = [0, 180]`，并运行真实 `Pose2Sim.poseEstimation()` 与 `Pose2Sim.triangulation()`。
- baseline TRC 和 rigid TRC 使用同一批 2D JSON、同一标定文件、同一帧范围。

## 根因与改动

- 原三角化逐个 marker 独立选择相机并独立估计 3D 点，`Hip/LHip/RHip` 与头面部 marker 的相对距离会逐帧变化。
- 新增 `triangulation.rigid_marker_groups`：配置后先按原流程得到 baseline，再从稳定帧估计组内刚体模板，逐帧联合最小化该组所有有效 2D 重投影误差。
- 对每个刚体组的 6DoF 变换参数做短窗口居中平滑，输出仍由同一个刚体模板变换得到，所以组内距离保持严格刚性。
- 这次只启用 `['Hip', 'LHip', 'RHip']` 和 `['Nose', 'LEye', 'REye', 'LEar', 'REar']`，没有对肩膀和脚部 marker 加约束。

## Before / After

- 髋部三点平均组内距离标准差：7.59 -> 0.00 mm。
- 头面部五点平均组内距离标准差：4.08 -> 0.00 mm。
- `Hip`: 7.72 -> 3.53 mm/frame² (-54.2%).
- `RHip`: 50.30 -> 3.69 mm/frame² (-92.7%).
- `LHip`: 38.62 -> 3.38 mm/frame² (-91.3%).

刚体拟合接受情况：
- `Hip+LHip+RHip`: accepted 180/180 frames, mean joint reprojection error 13.0 px.
- `Nose+LEye+REye+LEar+REar`: accepted 180/180 frames, mean joint reprojection error 8.5 px.

肩膀和脚部 marker 的抖动数值在本次实验中保持不变，这是预期结果：本次范围没有对这些 marker 施加刚体约束。

## Figures

![Hip trajectories](../figures/rigid_triangulation_demo0417v6/fig01_hip_trajectories.png)

![Pairwise stability](../figures/rigid_triangulation_demo0417v6/fig02_pairwise_distance_stability.png)

![Marker jitter](../figures/rigid_triangulation_demo0417v6/fig03_marker_jitter.png)

## Raw Evidence

- `data/rigid_triangulation_demo0417v6/baseline/demo0417v6_0-179_baseline.trc`
- `data/rigid_triangulation_demo0417v6/rigid/demo0417v6_0-179_rigid.trc`
- `data/rigid_triangulation_demo0417v6/pairwise_distance_metrics.csv`
- `data/rigid_triangulation_demo0417v6/marker_jitter_metrics.csv`
- `data/rigid_triangulation_demo0417v6/baseline_run.log`
- `data/rigid_triangulation_demo0417v6/rigid_run.log`

## Limits

- 本报告验证的是 demo0417v6 的前 180 帧。代码路径是真实 pipeline，但不是完整视频全帧。
- 这次没有更新下游 OpenSim marker/IK 设置。
- 如果还要直接降低肩膀或脚部 marker 抖动，需要另行配置肩带/足部刚体或半刚体组，风险和运动学约束需要单独评估。
