# 检测点抖动与三角化精度实验报告

## 实验设置

- 数据不是拍脑袋的截图，而是一个可控真值基准：240 帧、60 FPS、24 个随机种子。
- 2D 观测由平滑的 3D 真值轨迹投影得到，再加入高斯噪声、偶发大抖动、低置信度和缺失帧。
- 三角化直接调用仓库现有实现：`Pose2Sim.common.weighted_triangulation` 与 `Pose2Sim.triangulation.triangulation_from_best_cameras`。
- 去抖直接调用仓库现有实现：`Pose2Sim.filtering.filter1d`，2D 使用 OneEuro，3D 使用 Hampel + Butterworth。

## 结论先说

- 只做原始三角化时，3D RMSE 为 **4.27 +/- 0.46 mm**。
- 先做 2D OneEuro，再做 robust triangulation，最后做 3D Hampel + Butterworth 后，3D RMSE 降到 **2.07 +/- 0.26 mm**，3D 残余抖动降到 **0.44 +/- 0.09 mm**。
- 三角化策略从 4 相机无权重 DLT 换成 `robust_best_cams + 2D smoothing` 后，3D RMSE 从 **8.18 +/- 0.59 mm** 降到 **3.96 +/- 0.38 mm**。
- 同样是 robust triangulation，如果只剩 2 台相机，误差会上升到 **4.64 +/- 0.17 mm**；如果 4 台相机几乎都集中在同一侧，误差会上升到 **12.61 +/- 0.97 mm**。

## 回答问题 1：如何解决检测点看起来抖？

结论：不要只盯 3D 末端滤波，最有效的是 **先在 2D 上做轻量时序平滑，再进入 triangulation**。

原因：
- 视觉上看到的“抖”，本质是逐帧 2D 关键点高频噪声。
- 如果等到 3D 再滤，三角化已经把多视角噪声耦合进去了；后滤只能补救，不能从源头减少误差。
- `OneEuro` 对这种抖动很合适，因为它在低速段更强抑噪，在快速运动时又不会像重滤波那样把运动幅度压扁。

建议顺序：
1. 2D 先做 OneEuro。
2. triangulation 继续用现有 robust best cameras。
3. 3D 再做 Hampel + Butterworth 作为收尾。

![抖动示例](../figures/jitter_triangulation/fig01_jitter_example.png)

![去抖消融](../figures/jitter_triangulation/fig02_jitter_ablation.png)

## 回答问题 2：如何进一步提高三角化精度？

从这组代码实验看，提升顺序基本是：

1. **先提升 2D 输入质量**：2D smoothing 对 3D RMSE 和抖动都直接有帮助。
2. **保留 confidence weighting**：有权重比无权重更稳，因为低质量视角会自动被降权。
3. **对低置信度点做阈值剔除**：把明显坏的视角先排掉，比把它们硬塞进 DLT 更有效。
4. **继续使用 robust camera exclusion**：当某个相机瞬时飘掉时，`triangulation_from_best_cameras` 明显比“所有相机都参与”更稳。
5. **尽量保留 3 台以上并拉开视角张角**：这是几何层面的硬收益，代码补救不了同侧聚集视角的深度病态。

![三角化消融](../figures/jitter_triangulation/fig03_triangulation_ablation.png)

## 推荐落地方案

如果你现在要改自己的工程，我建议按下面优先级来：

- `pose` 阶段新增一个 2D temporal smoothing 开关，优先接 OneEuro。
- `triangulation.likelihood_threshold_triangulation` 保持在 0.3 左右起步，再按数据调。
- 保留 `triangulation_from_best_cameras` 这类基于重投影误差的相机剔除逻辑。
- `filtering` 阶段开启 `reject_outliers = true` 和 `type = 'butterworth'` 作为默认后处理。
- 采集端优先保证更多视角和更大的视角张角，而不是只靠后处理。

## 原始结果文件

- `data/jitter_triangulation/jitter_metrics_raw.csv`
- `data/jitter_triangulation/triangulation_metrics_raw.csv`
- `data/jitter_triangulation/jitter_metrics_summary.csv`
- `data/jitter_triangulation/triangulation_metrics_summary.csv`
