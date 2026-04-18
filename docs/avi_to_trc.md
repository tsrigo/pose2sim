# AVI 到 3D TRC 的批处理命令行工具

`avi_to_trc` 是一个独立的 Pose2Sim 命令行工具。它面向 Pose2Sim 风格的单个 trial，读取多机位 `*.avi` 视频，运行 RTMLib 2D 姿态估计，写出 OpenPose 风格 JSON，然后复用 Pose2Sim 现有三角化流程生成标准 3D `.trc`。

## 它解决什么问题

- 输入：一个 trial 目录，里面有 `videos/*.avi`
- 2D 推理：逐帧检测 + 批量 RTMPose crop 推理
- 中间产物：`pose/<camera>_json/*.json`
- 最终输出：`pose-3d/*.trc`

如果你的目标只是“从 AVI 跑到 TRC”，而不关心后续 kinematics，这个工具就是直接路径。

## 目录约定

命令要求目录满足 Pose2Sim 的 trial 结构：

```text
Session/
  Config.toml            # 如果 Trial_1/Config.toml 存在，这个可以选填
  calibration/
    *.toml               # 三角化前必须已有标定文件
  Trial_1/
    Config.toml          # 如果 Session/Config.toml 存在，这个可以选填
    videos/
      cam01.avi
      cam02.avi
      cam03.avi
      cam04.avi
```

几个关键约束：

- trial 下必须有 `videos/*.avi`
- session 的 `calibration/` 下必须已经有 `.toml` 标定文件
- v1 只支持单人场景
- v1 只支持 top-down RTMLib 模型，也就是“detector + pose model”两阶段结构
- v1 到生成 TRC 为止，不包含后续 kinematics

## 基本用法

```bash
avi_to_trc --trial-dir /path/to/Session/Trial_1 --batch-size 16 --overwrite-pose
```

常用参数：

- `--config /path/to/Config.toml`
- `--batch-size 16`
- `--det-frequency 4`
- `--backend onnxruntime`
- `--device cuda`
- `--overwrite-pose`

如果不传 `--config`，配置文件按下面顺序解析：

1. `Trial_1/Config.toml` 覆盖 `Session/Config.toml`
2. `Trial_1/Config.toml`
3. `Session/Config.toml`

## 端到端示例

如果标定已经存在，可以直接运行：

```bash
avi_to_trc \
  --trial-dir /data/my_session/Trial_1 \
  --batch-size 16 \
  --backend onnxruntime \
  --device cuda \
  --overwrite-pose
```

运行后会生成：

- `Trial_1/pose/cam01_json/*.json`
- `Trial_1/pose/cam02_json/*.json`
- `Trial_1/pose/cam03_json/*.json`
- `Trial_1/pose/cam04_json/*.json`
- `Trial_1/pose-3d/<trial_name>_<start>-<end>.trc`

## 这条路径和标准 Pose2Sim 有什么关系

这条 CLI 本质上是：

1. 直接读取 `AVI`
2. 跑批量 2D 姿态推理
3. 导出 OpenPose JSON
4. 直接调用 Pose2Sim 三角化生成 TRC

也就是说，它走的是“AVI -> JSON -> TRC”，而不是完整的 `runAll()`。

## 单人场景下是否需要 `personAssociation`

这条 CLI 的 v1 设计就是单人路径，因此不会再单独跑 `personAssociation`。

原因是：

- 它只支持单人场景
- 输出 JSON 会直接交给三角化
- 对干净的单人实验，通常不需要额外的人物关联阶段

如果场景里可能有路人、误检或多人重叠，那么更稳妥的做法仍然是走标准 Pose2Sim 流程，并保留 `personAssociation`。

## 性能模型

当前的 batching 主要发生在 pose 推理阶段，不一定发生在 detection 阶段。

原因是默认轻量 YOLOX detector 的 ONNX 输入往往是固定 `batch=1`，例如 `[1, 3, H, W]`。这会导致：

- detection 仍然逐帧执行
- RTMPose crop 可以批量执行

因此在实际运行中：

- `batch_size=1` 基本等同顺序 crop 推理
- 更大的 `batch_size` 只有在 pose 模型能吃 batch 时才会提升吞吐
- `det_frequency` 仍然决定 detector 何时刷新 bbox

这也解释了为什么有时把 `batch_size` 调大了，但 GPU 占比和总时间变化并不明显。

## 输出语义

这个工具产出的 `.trc` 是标准的 3D OpenSim 兼容 TRC，由 Pose2Sim 现有三角化模块生成。  
它不会生成假的 2D TRC，也不会生成单目伪 3D。

## 常见报错

`No calibration TOML file found`

- 说明还没有把标定转换或生成成 `.toml`，先补齐标定文件。

`At least two camera JSON streams are required`

- 三角化至少需要两路同步相机。

`avi_to_trc only supports top-down RTMLib models`

- 需要切换到 detector + RTMPose 的两阶段模型，比如内置的 `Body_with_feet`、`Whole_body`、`Body`。

`Existing pose-sync/pose-associated JSON outputs`

- 说明该 trial 下已有旧的下游 JSON 结果。可以加 `--overwrite-pose` 重新生成，或者先清理旧目录。

## 使用说明

- 该 CLI 会强制 `output_format = 'openpose'`，因为三角化阶段消费的是 OpenPose 风格 JSON。
- 如果多路相机视频长度不一致，它会自动裁到所有相机都有效的公共帧区间，并打印警告。
- 如果已经存在兼容的 pose JSON，而你没有传 `--overwrite-pose`，它会优先复用现有结果。
