# Pose2Sim demo0417v6 刚体组三角化交付包说明

## 这次改了什么

- 在 `Pose2Sim/triangulation.py` 增加可配置的刚体 marker 组：
  - `["Hip", "LHip", "RHip"]`
  - `["Nose", "LEye", "REye", "LEar", "REar"]`
- 原流程仍然先逐点三角化；启用 `triangulation.rigid_marker_groups` 后，会从稳定帧估计组内模板，再逐帧联合最小化整组 marker 的 2D 重投影误差。
- 输出默认不再 100% 替换成刚体模板，而是通过 `rigid_group_blend` 混回 baseline，并用最大修正量与 pairwise 距离变化保护髋部比例。
- 修复 Windows/Anaconda 环境下 `parallel_triangulation` 并行三角化的 pickle 报错：`toml` inline table 会生成不可序列化的 `DynamicInlineTableDict`，现在只在传入 worker process 前递归转成普通 `dict/list/tuple`。
- 默认不影响旧项目；只有配置 `rigid_marker_groups` 时才启用。

## 怎么启用

在 `Config.toml` 的 `[triangulation]` 下加入：

```toml
rigid_marker_groups = [
  ["Hip", "LHip", "RHip"],
  ["Nose", "LEye", "REye", "LEar", "REar"]
]
rigid_group_smoothing_window = 5
rigid_group_blend = 0.7
rigid_group_max_correction_m = 0.05
rigid_group_max_pairwise_change_ratio = 0.15
rigid_group_fill_missing = false
rigid_group_max_cams_to_exclude = 1
rigid_group_max_nfev = 30
```

## 真实数据验证结果

- 数据：用户提供的 `data/demo0417v6.zip`。
- 验证范围：四相机可同步的完整范围 `frame_range = [0, 391]`，共 391 帧，30 fps，约 13.03 秒。
- 实际运行路径：`Pose2Sim.poseEstimation()` 生成 2D JSON，然后使用同一批 2D JSON 分别跑 baseline 与刚体组三角化。
- 并行验证：同一真实 demo 配置在 `parallel_triangulation = 2` 下跑通 `frame_range = [0, 20]`，确认不再需要临时改成 `parallel_triangulation = false` 来绕过 Windows pickle 报错。

关键结果：

- 髋部三点平均组内距离标准差：`7.81 -> 2.18 mm`
- 髋部 pairwise 距离相对 baseline 的最大帧级变化：`p95 <= 14.0%`, `max <= 15.0%`
- 头面部五点平均组内距离标准差：`4.29 -> 1.47 mm`
- `Hip` 抖动：`7.60 -> 7.03 mm/frame²`，下降 `7.5%`
- `RHip` 抖动：`58.49 -> 23.09 mm/frame²`，下降 `60.5%`
- `LHip` 抖动：`41.58 -> 15.68 mm/frame²`，下降 `62.3%`
- 两个刚体组都接受 `391/391` 帧。

## 可视化结果

可直接放入 OpenSim / 支持 TRC 的可视化工具：

- 修改前：`data/rigid_triangulation_demo0417v6_full/baseline/demo0417v6_0-390_baseline.trc`
- 修改后：`data/rigid_triangulation_demo0417v6_full/rigid/demo0417v6_0-390_rigid.trc`

完整报告见：

- `docs/rigid-triangulation-demo0417v6-full.md`

## 范围限制

- 这次只约束髋部三点和头面部五点，没有对肩膀和脚部 marker 加刚体约束，因此肩膀和脚部的 TRC 数值基本保持不变。
- `cam01/cam03/cam04` 有 467 帧，`cam02` 有 391 帧；报告使用四相机共同可用的 391 帧范围。
- 没有修改下游 OpenSim marker/IK 配置。
