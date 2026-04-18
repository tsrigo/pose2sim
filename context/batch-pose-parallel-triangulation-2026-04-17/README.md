# 批量姿态推理与并行三角化上下文

该目录记录本次“批量姿态推理 + 并行三角化”里程碑的本地实现证据。

它是给 PR 描述和代码审阅使用的本地材料，不是面向最终用户的入门文档。

## 本次 PR 包含的范围

- `Pose2Sim/poseEstimation.py`
- `Pose2Sim/triangulation.py`
- `Pose2Sim/Demo_SinglePerson/Config.toml`
- `Pose2Sim/Demo_MultiPerson/Config.toml`
- `Pose2Sim/Demo_Batch/Config.toml`
- `Pose2Sim/Demo_Batch/Trial_1/Config.toml`
- `Pose2Sim/Demo_Batch/Trial_2/Config.toml`
- `docs/README.md`
- `docs/batch-pose-and-parallel-triangulation.md`
- `context/batch-pose-parallel-triangulation-2026-04-17/*`

## 明确排除在本次 PR 范围外的本地改动

这些文件虽然当时存在于工作树里，但不属于本里程碑：

- `pyproject.toml`
- `Pose2Sim/Utilities/avi_to_trc.py`
- `data/`

## 环境

- 日期：`2026-04-17`
- PR 分支创建前的仓库提交：`614804c`
- Python：`3.13.5`
- onnxruntime：`1.24.4`
- 本地可见 provider：`['TensorrtExecutionProvider', 'CUDAExecutionProvider', 'CPUExecutionProvider']`
- OpenCV：`4.13.0`

## 静态验证

执行命令：

```bash
python -m py_compile Pose2Sim/poseEstimation.py Pose2Sim/triangulation.py
python - <<'PY'
import Pose2Sim.poseEstimation
import Pose2Sim.triangulation
print("imports_ok")
PY
```

结果：

- 编译成功
- 模块导入成功

## ONNX Batch 能力探测

命令摘要：

- 在 `onnxruntime` + CPU 上实例化 `BodyWithFeet` 轻量模型
- 检查 detector 和 pose 模型 `session.get_inputs()[0].shape`

观测到的输入形状：

- detector 输入：`[1, 3, 416, 416]`
- pose 输入：`['batch', 3, 256, 192]`

解释：

- 本地这次实跑用到的轻量 detector 不支持动态 batch
- 轻量 pose 模型支持动态 batch
- 因此预期行为是“detector 回退为顺序执行，pose crop 仍然批量执行”

## 本地运行验证

验证项目：

- 将 `Pose2Sim/Demo_SinglePerson` 复制到一个临时目录
- 只处理 `0..3` 帧
- 强制使用 `backend = 'onnxruntime'`、`device = 'cpu'`、`display_detection = false`、`save_video = 'none'`

### Pose Estimation 运行

配置覆盖：

- `mode = 'lightweight'`
- `det_frequency = 1`
- `batch_size = 2`
- `parallel_workers_pose = 1`
- `overwrite_pose = true`

结果：

- `Pose2Sim.poseEstimation(config)` 运行成功
- `cam01`、`cam02`、`cam03`、`cam04` 各生成 `4` 个 JSON
- 这次 CPU-only 本地验证的总耗时为 `4.236 s`

关键日志：

```text
Inference run on every single frame.
GPU pose batching requested with batch_size=2.
Detection ONNX model has fixed batch=1, falling back to sequential inference for this stage.
```

含义：

- 新的 batch 路径确实进入了
- detector 的固定 batch 回退保护在真实模型上被触发了
- 即使 detector 不能 batch，pose estimation 仍成功完成并产出了结果

### Triangulation 运行

配置覆盖：

- `parallel_triangulation = 2`
- `min_chunk_size = 1`
- `make_c3d = false`

额外说明：

- 为了兼容现有标定查找逻辑，临时 demo 被包在一个符合 session 结构的父目录下

结果：

- `Pose2Sim.triangulation(config)` 运行成功
- 日志出现 `Triangulating frames in parallel with 2 worker processes.`
- 生成 TRC：`Demo_SinglePerson_0-3.trc`
- 这次 CPU-only 本地验证的总耗时为 `1.689 s`

关键日志：

```text
Triangulating frames in parallel with 2 worker processes.
3D coordinates are stored at .../pose-3d/Demo_SinglePerson_0-3.trc.
```

## 备注与限制

- 这份上下文证明了新的 batch 路径和并行三角化路径在真实本地运行中都能成功执行。
- 它还不能证明 GPU 上一定已经拿到了显著加速比。
- 用户提出的真实验证目标 `data/20260402/recordings/PNV-ITM-001` 没有包含在这次 PR 的验证范围内。
- 这次 CPU-only demo 暴露出一个关键兼容性事实：实际是否提速，很大程度取决于 detector ONNX 模型是否支持动态 batch。当前实现已经能在 fixed-batch detector 上安全回退，而不是直接失败。
