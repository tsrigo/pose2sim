# Pose2Sim 修改文档：GPU Batch 推理 & 单人跳过 PersonAssociation

## 一、修改概述

本次修改包含两个功能：

1. **GPU Batch 推理**：`poseEstimation.py` 支持多帧同时送入 GPU 进行检测和姿态估计，通过 `batch_size` 配置项控制
2. **单人跳过 PersonAssociation**：当 `multi_person = false` 时，自动跳过 `personAssociation()` 步骤

## 二、修改文件清单

| 文件 | 修改内容 |
|------|----------|
| `Pose2Sim/poseEstimation.py` | 新增 `_batch_onnx_inference()`、`batch_detection()`、`batch_pose_topdown()`、`process_video_batched()`；修改 `estimate_pose_all()` 读取 `batch_size` 并分发 |
| `Pose2Sim/Pose2Sim.py` | `personAssociation()` 方法增加单人跳过逻辑 |
| `Pose2Sim/Demo_SinglePerson/Config.toml` | 新增 `batch_size = 1` |
| `Pose2Sim/Demo_MultiPerson/Config.toml` | 新增 `batch_size = 1` |
| `Pose2Sim/Demo_Batch/Config.toml` | 新增 `batch_size = 1` |

---

## 三、Feature 1：GPU Batch 推理

### 3.1 设计原理

RTMLib 的 ONNX 模型原生支持动态 batch 维度，但 `rtmlib` 的 `BaseTool.inference()` 硬编码 `batch=1`：
```python
# rtmlib/tools/base.py line 129
input = img[None, :, :, :]  # 始终 (1, 3, H, W)
```

本修改**不修改 rtmlib**，而是在 Pose2Sim 层面直接访问 ONNX session，绕过 batch=1 限制。

### 3.2 新增函数

#### `_batch_onnx_inference(session, batch_input)`
- 直接调用 `session.run()`，输入 `(N, 3, H, W)` 张量
- 绕过 rtmlib 的单帧限制

#### `batch_detection(det_model, frames)`
- 对 N 帧逐帧调用 `det_model.preprocess()` 预处理
- `np.stack` 拼成 `(N, 3, 640, 640)` batch 张量
- 单次 ONNX 调用完成所有帧的检测
- 逐帧调用 `det_model.postprocess()` 解码 bbox

#### `batch_pose_topdown(pose_model, frames, per_frame_bboxes, max_batch_crops=128)`
- 收集所有帧的所有 bbox crop，记录 crop→帧 的映射关系
- 当 crop 总数超过 `max_batch_crops` 时自动分 sub-batch，防止 GPU OOM
- 利用 `get_simcc_maximum()` 原生支持 `(N, K, Wx)` 的特性做向量化后处理
- 按帧分组返回结果

#### `process_video_batched(...)`
- 每次读取 `batch_size` 帧
- 处理 `det_frequency`：仅对 `frame_idx % det_frequency == 0` 的帧做检测，其余复用缓存 bbox
- batch 检测 → batch 姿态估计 → 逐帧 NMS/tracking/person_filter（必须顺序执行）

### 3.3 配置方式

```toml
[pose]
det_frequency = 4
batch_size = 4    # 同时处理 4 帧，需要更多显存但速度更快
device = 'cuda'
backend = 'onnxruntime'
```

- `batch_size = 1`（默认）：行为与修改前完全一致
- `batch_size > 1`：仅在 `backend = 'onnxruntime'` 且 top-down 模型时生效
- 不满足条件时自动 fallback 到单帧处理，并输出 warning

### 3.4 向后兼容性

- 旧的 `Config.toml` 没有 `batch_size` 字段时，默认值为 1，行为不变
- 单帧路径（`process_video`）代码完全未修改

---

## 四、Feature 2：单人跳过 PersonAssociation

### 4.1 设计原理

Pose2Sim 文档说明单人场景不需要运行 `personAssociation()`。`triangulation.py` 已有 fallback 链：

```
pose-associated/ → pose-sync/ → pose/
```

当 `pose-associated/` 不存在时，triangulation 自动从 `pose/` 读取。因此跳过 personAssociation 不需要修改 triangulation。

### 4.2 实现

`Pose2Sim.py` 的 `personAssociation()` 方法：

```python
def personAssociation(self):
    from Pose2Sim.personAssociation import associate_all
    for config_dict in self.config_dicts:
        multi_person = config_dict.get('project', {}).get('multi_person', False)
        if not multi_person:
            logging.info('Single-person mode: skipping person association. '
                        'Triangulation will read from pose/ or pose-sync/ directly.')
            continue
        # ... 原有逻辑不变
```

---

## 五、正确性验证

### 测试 1：`_batch_onnx_inference` 不同 batch size

使用 Mock ONNX session 验证 batch=1、4、8 时输出形状正确。

```
PASS: _batch_onnx_inference batch=1  → output shape (1, 26, 384)
PASS: _batch_onnx_inference batch=4  → output shape (4, 26, 384)
PASS: _batch_onnx_inference batch=8  → output shape (8, 26, 384)
```

### 测试 2：`batch_pose_topdown` 多帧多人

4 帧，分别有 1/2/1/0 个人（0 人时 fallback 到全帧 bbox）：

```
PASS: Frame 0: kpts=(1, 26, 2), scores=(1, 26)
PASS: Frame 1: kpts=(2, 26, 2), scores=(2, 26)
PASS: Frame 2: kpts=(1, 26, 2), scores=(1, 26)
PASS: Frame 3: kpts=(1, 26, 2), scores=(1, 26)  ← 全帧 fallback
```

### 测试 3：`batch_detection` 多帧检测

4 帧 batch 检测，每帧 2 个有效检测框：

```
PASS: Frame 0: 2 detections
PASS: Frame 1: 2 detections
PASS: Frame 2: 2 detections
PASS: Frame 3: 2 detections
PASS: Empty input handled correctly
```

### 测试 4：Sub-batching 防 OOM

10 帧 × 5 人 = 50 crops，`max_batch_crops=16` 时应分 4 次 ONNX 调用：

```
ONNX session.run() called 4 times for 50 crops (max_batch_crops=16)
PASS: Sub-batching works correctly (4 sub-batches)
PASS: All frames got correct number of persons after sub-batching
```

### 测试 5：PersonAssociation 跳过逻辑

```
PASS: multi_person=false -> SKIPPED
PASS: multi_person=true  -> RAN
```

### 测试 6：Config 向后兼容

```
PASS: batch_size present -> 8
PASS: batch_size absent  -> 1 (backward compatible)
```

### 测试 7：Batch vs 单帧结果一致性（关键）

验证 `get_simcc_maximum` 的 batch 处理与逐帧处理输出完全相同：

```
PASS: Batch vs single get_simcc_maximum results are identical
  Max location diff: 0.00e+00
  Max score diff:    0.00e+00
```

### 测试 8：Batch 坐标逆变换一致性（关键）

验证 batch 后处理（SimCC 解码 + 坐标逆变换）与 RTMPose 原始单帧 postprocess 的数值一致：

```
PASS: Batch rescale matches single-frame rescale (max diff: 0.00e+00)
```

**精度完全相同**，无浮点误差。这是因为所有运算（argmax、除法、乘法、加法）都是确定性的。

### 测试 9：Triangulation Fallback 链验证

模拟 `pose-associated/` 不存在时，triangulation 的目录查找逻辑：

```
PASS: Without pose-associated/, triangulation falls back to pose/
  Found 2 cameras with [1, 1] files each
```

### 测试 10：Python 语法检查

```
$ python -m py_compile Pose2Sim/poseEstimation.py  → OK
$ python -m py_compile Pose2Sim/Pose2Sim.py        → OK
```

### 测试 11：Import 验证

```python
from Pose2Sim.poseEstimation import _batch_onnx_inference, batch_detection, batch_pose_topdown, process_video_batched
# All batch functions imported successfully
```

---

## 六、使用建议

### Batch 大小选择

| batch_size | 显存占用（估算，HALPE_26 balanced） | 适用场景 |
|------------|--------------------------------------|----------|
| 1 | ~1 GB | 默认，低显存 |
| 4 | ~2 GB | 推荐起步值 |
| 8 | ~3 GB | 8GB 显卡 |
| 16 | ~5 GB | 12GB+ 显卡 |

实际显存取决于视频分辨率、每帧检测人数、模型大小。

### 单人流程简化

```python
from Pose2Sim import Pose2Sim

Pose2Sim.calibration()
Pose2Sim.poseEstimation()      # batch_size=4 in Config.toml
Pose2Sim.synchronization()
Pose2Sim.personAssociation()   # 自动跳过 (multi_person=false)
Pose2Sim.triangulation()       # 自动从 pose/ 或 pose-sync/ 读取
Pose2Sim.filtering()
Pose2Sim.kinematics()
```

即使调用了 `personAssociation()`，单人模式下也会立即返回，不浪费时间。
