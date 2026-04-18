# 批量姿态推理与并行三角化

本文说明 Pose2Sim 在 `poseEstimation()` 和 `triangulation()` 两个阶段新增的吞吐优化选项，以及它们什么时候真的会提速，什么时候看起来“开了也没效果”。

## 实现了什么

Pose estimation 侧新增或打通了三层能力：

- `pose.batch_size`：把多帧检测输入和多人 crop 输入尽量合并后再做 ONNX 推理。
- `pose.parallel_workers_pose`：按视频文件并行，每个 worker 处理一个视频。
- `pose.det_frequency`：仍然控制隔多少帧跑一次 detector，但会影响 batching 的收益上限。

Triangulation 侧新增了一层能力：

- `triangulation.parallel_triangulation`：按帧并行三角化，底层使用 `ProcessPoolExecutor`。

## 推荐配置

建议在 `Config.toml` 中使用下面这一组起步配置：

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

GPU 上的推荐起点：

- `batch_size = 8`
- `parallel_workers_pose = 'auto'`
- `det_frequency = 1`
- `parallel_triangulation = 'auto'`

## Pose 侧现在怎么跑

当 `batch_size > 1` 时，Pose2Sim 会从原来的逐帧路径切到批处理视频路径。

高层流程如下：

1. 一次读取最多 `batch_size` 帧。
2. 如果 detector 的 ONNX 输入支持动态 batch，就把这些帧一起送进 detector。
3. 收集这批帧里的所有人体 crop，一次性送进 pose 模型。
4. 跟踪、NMS、JSON 导出、可视化仍按帧顺序执行。

这样做的目的不是改输出格式，而是把最重的 ONNX 推理尽量并成大调用。

## `parallel_workers_pose` 到底有没有实现

有，且当前实现是“按视频并行”，不是“单个视频内部按帧并行”。

这意味着：

- 如果一个 trial 里有 4 个相机视频，那么它们可以被不同 worker 同时处理。
- 如果只有 1 个视频，`parallel_workers_pose` 基本不会带来收益。
- 如果 `display_detection = true`，代码会主动退回顺序执行。

因此，老师看到“开了 `parallel_workers_pose` 也没明显提速”，并不等于它没实现，更常见的原因是：

- 测试场景本来就不满足它的收益条件；
- GPU 主要瓶颈不在按视频并行，而在 detector 仍然是单帧；
- 单人、短视频、低分辨率时，每个 worker 本身的活不够重。

## 为什么 `batch_size` 可能几乎不提速

这是这次实现里最关键的限制。

实际是否提速，取决于 detector 和 pose 两个 ONNX 模型是否都支持动态 batch，而且它们是分别检查的。

典型情况有四种：

- detector 动态 batch，pose 也动态 batch：两边都能 batch，收益最大。
- detector 固定 batch，pose 动态 batch：detector 仍逐帧，但 pose crop 还能 batch。
- detector 动态 batch，pose 固定 batch：只有 detector 受益。
- 两者都固定 batch：逻辑仍正确，但速度基本接近旧路径。

当前本地实跑里，轻量 detector 的输入形状是 `[1, 3, 416, 416]`，也就是固定 batch=1；pose 模型输入是 `['batch', 3, 256, 192]`，也就是 pose 可以 batch。  
这正好解释了“`batch_size` 变大但 GPU 占比还是小、速度也没明显提升”的现象：真正吃算力的那一部分并没有完全放大。

## `det_frequency` 为什么会影响收益

- 当 `det_frequency = 1` 时，detector 和 pose 都更容易跨帧做完整 batch。
- 当 `det_frequency > 1` 时，代码会保留旧逻辑里的 bbox 传播语义，也就是 detector 帧之间的 bbox 状态仍带有顺序依赖。

这时仍然可以：

- 对 detector 真正触发的那些帧做批量检测；
- 对单帧内或局部帧组中的 pose crop 做批量推理。

但 detector 侧的收益会明显变小。

## 三角化并行做了什么

`parallel_triangulation` 用于按帧并行执行纯三角化计算。

- `'auto'`：最多使用 `os.cpu_count()` 个进程，并受帧数上限约束。
- 整数：使用指定数量的 worker。
- `false`：关闭该并行路径。

需要注意的一点：

- 多人场景下，跨帧的人物重排序仍然保持顺序执行。

这是为了保证身份连续性不被进程并行打乱。

## 单人场景还要不要 `personAssociation`

如果场景里确实只有一个人，而且每路相机每帧都只检测到这个人，那么通常可以跳过 `personAssociation`。

原因是三角化阶段会按以下顺序找输入：

1. `pose-associated`
2. `pose-sync`
3. `pose`

也就是说，没有做 `personAssociation` 时，三角化仍然可以直接消费 `pose/*.json`。

但如果虽然是单人实验，画面里偶尔有路人、误检或多框，那么保留 `personAssociation` 依然有价值，因为它能帮助筛出真正的目标人。

另外，`avi_to_trc` 这条新 CLI 在 v1 里就是单人路径，会直接从 `pose` JSON 走到最终 TRC，不包含 `personAssociation`。

## 什么时候最容易看到提速

更容易看到收益的条件是：

- `backend = 'onnxruntime'`
- `device = 'cuda'`
- `display_detection = false`
- `det_frequency = 1`
- 多相机或多视频同时处理
- 每帧可见人体数较多，或者分辨率较高

反过来，以下情况即使开了也可能不明显：

- detector ONNX 固定 batch=1
- 单人、短片段、低分辨率
- 只测 1 路视频
- `display_detection = true`
- `det_frequency > 1`

## 部署建议

第一次上真实 GPU 数据时，建议按下面顺序测：

1. 关闭实时显示。
2. 设 `batch_size = 8`。
3. 先用 `det_frequency = 1` 测完整 batch 路径。
4. 打开 `parallel_triangulation = 'auto'`。
5. 在一小段数据上对比时间和输出，再放大到全量。

如果显存吃紧：

- 把 `batch_size` 从 `8` 降到 `4`
- 保持 `parallel_workers_pose` 较保守
- 稳定后再增大

## 输出兼容性

这次改动不改变输出契约：

- 2D 结果仍然写成 OpenPose 风格 JSON
- 跟踪顺序仍按帧生效
- 三角化仍输出相同格式的 TRC

它解决的是吞吐问题，不是文件格式问题。
