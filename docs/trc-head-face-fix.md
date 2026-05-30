# TRC 耳朵与头顶点修复报告

## 结论

- 我已经实际跑了修复前和修复后的 `Pose2Sim.triangulation.triangulate_all()`，输入保持一致，确认问题存在且修复有效。
- 根因不是三角化数值算法把点算丢了，而是 `Pose2Sim/skeletons.py` 里 `COCO`、`BODY_25`、`BODY_25B`、`BODY_135` 的 skeleton 定义漏掉了官方已有的 `REye/LEye/REar/LEar`，而 `BODY_25B` 与 `BODY_135` 还没有把官方的 `HeadTop` 暴露到 TRC。
- 修复后，同一份 2D JSON 重新三角化：
  - `COCO` / `BODY_25`：TRC 新增双眼和双耳。
  - `BODY_25B` / `BODY_135`：TRC 新增双眼、双耳，并新增 `HeadTop`；同时保留原来的 `Head` 兼容旧流程。

## 复现实验

- 输入：四机位、两帧、合成标定、合成 2D JSON。JSON 里显式包含头部相关点。
- 运行入口：仓库现有 `Pose2Sim.triangulation.triangulate_all()`。
- 比较对象：完全相同的输入，在修复前后各跑一次。

### 图 1：修复前后关键点存在性

![Presence](../figures/trc_head_face_fix/fig01_presence_before_after.png)

### 图 2：TRC marker 数量变化

![Counts](../figures/trc_head_face_fix/fig02_marker_count.png)

### 图 3：BODY_25B / BODY_135 表头前后对比

![Headers](../figures/trc_head_face_fix/fig03_header_diff.png)

## 数值结果

| pose_model   |   before_markers |   after_markers | before_REar   | after_REar   | before_HeadTop   | after_HeadTop   |
|:-------------|-----------------:|----------------:|:--------------|:-------------|:-----------------|:----------------|
| COCO         |               14 |              18 | no            | yes          | no               | no              |
| BODY_25      |               21 |              25 | no            | yes          | no               | no              |
| BODY_25B     |               21 |              26 | no            | yes          | no               | yes             |
| BODY_135     |               27 |              32 | no            | yes          | no               | yes             |

| row             |   REye |   LEye |   REar |   LEar |   Head |   HeadTop |
|:----------------|-------:|-------:|-------:|-------:|-------:|----------:|
| COCO before     |      0 |      0 |      0 |      0 |      0 |         0 |
| COCO after      |      1 |      1 |      1 |      1 |      0 |         0 |
| BODY_25 before  |      0 |      0 |      0 |      0 |      0 |         0 |
| BODY_25 after   |      1 |      1 |      1 |      1 |      0 |         0 |
| BODY_25B before |      0 |      0 |      0 |      0 |      1 |         0 |
| BODY_25B after  |      1 |      1 |      1 |      1 |      1 |         1 |
| BODY_135 before |      0 |      0 |      0 |      0 |      1 |         0 |
| BODY_135 after  |      1 |      1 |      1 |      1 |      1 |         1 |

| pose_model   |   before |   after |
|:-------------|---------:|--------:|
| COCO         |       14 |      18 |
| BODY_25      |       21 |      25 |
| BODY_25B     |       21 |      26 |
| BODY_135     |       27 |      32 |

修复前后的原始 TRC 已保存：

- `data/trc_head_face_fix/before/*.trc`
- `data/trc_head_face_fix/after/*.trc`

## Debug 过程

- `triangulation.py` 用 `RenderTree(model)` 生成 `keypoints_ids` / `keypoints_names`，TRC 表头完全由这两个列表决定。
- 因此只要 skeleton 没有定义某个节点，即使 2D JSON 里有该 triplet，三角化也不会读取，更不会写进 TRC。
- 修复前：
  - `BODY_25` 没有 `REye/LEye/REar/LEar`。
  - `BODY_25B` 和 `BODY_135` 没有 `REye/LEye/REar/LEar/HeadTop`，并把 id 18 只暴露成 `Head`。
- 修复方式：只改 `Pose2Sim/skeletons.py`，把官方已有 body-part 映射补进 tree；没有改三角化数值过程。

代码 diff 摘录：

```diff
diff --git a/Pose2Sim/skeletons.py b/Pose2Sim/skeletons.py
index 71e856e..1ca5e4b 100644
--- a/Pose2Sim/skeletons.py
+++ b/Pose2Sim/skeletons.py
@@ -700,8 +700,13 @@ BODY_25B = Node("CHip", id=None, children=[
         ]),
     ]),
     Node("Neck", id=17, children=[
-        Node("Head", id=18, children=[
-            Node("Nose", id=0),
+        Node("Head", id=18),
+        Node("HeadTop", id=18),
+        Node("Nose", id=0, children=[
+            Node("REye", id=2),
+            Node("LEye", id=1),
+            Node("REar", id=4),
+            Node("LEar", id=3),
         ]),
         Node("RShoulder", id=6, children=[
             Node("RElbow", id=8, children=[
@@ -741,7 +746,12 @@ BODY_25 = Node("CHip", id=8, children=[
         ]),
     ]),
     Node("Neck", id=1, children=[
-        Node("Nose", id=0),
+        Node("Nose", id=0, children=[
+            Node("REye", id=15),
+            Node("LEye", id=16),
+            Node("REar", id=17),
+            Node("LEar", id=18),
+        ]),
         Node("RShoulder", id=2, children=[
             Node("RElbow", id=3, children=[
                 Node("RWrist", id=4),
@@ -780,8 +790,13 @@ BODY_135 = Node("CHip", id=None, children=[
         ]),
     ]),
     Node("Neck", id=17, children=[
-        Node("Head", id=18, children=[
-            Node("Nose", id=0),
+        Node("Head", id=18),
+        Node("HeadTop", id=18),
+        Node("Nose", id=0, children=[
+            Node("REye", id=2),
+            Node("LEye", id=1),
+            Node("REar", id=4),
+            Node("LEar", id=3),
         ]),
         Node("RShoulder", id=6, children=[
             Node("RElbow", id=8, children=[
@@ -957,7 +972,12 @@ COCO = Node("CHip", id=None, children=[
         ]),
     ]),
     Node("Neck", id=1, children=[
-        Node("Nose", id=0),
+        Node("Nose", id=0, children=[
+            Node("REye", id=14),
+            Node("LEye", id=15),
+            Node("REar", id=16),
+            Node("LEar", id=17),
+        ]),
         Node("RShoulder", id=2, children=[
             Node("RElbow", id=3, children=[
                 Node("RWrist", id=4),
```

## 官方映射依据

- OpenPose BODY_25 官方输出顺序文档：https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md
- OpenPose `BODY_25B` / `BODY_135` 官方 body-part 映射源码：https://huggingface.co/camenduru/openpose/blob/5e17f6ad43ab415a0114537541a8d37d2503424f/src/openpose/pose/poseParameters.cpp

从官方映射可以直接看到：

- `BODY_25`：15/16/17/18 分别是 `REye/LEye/REar/LEar`。
- `BODY_25B`：1/2/3/4/18 分别是 `LEye/REye/LEar/REar/HeadTop`。
- `BODY_135`：同样把 1/2/3/4/18 定义为 `LEye/REye/LEar/REar/HeadTop`。

## 影响范围

- 这次修复直接影响 TRC 列名与导出的 3D 点集合。
- `BODY_25B` 和 `BODY_135` 为了兼容现有 OpenSim 配置，继续保留了 `Head`，同时新增同坐标的 `HeadTop`。
- 我没有改 OpenSim 的 marker XML / IK setup，所以这次提交聚焦在“TRC 缺点”问题本身。
