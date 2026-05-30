# 2D 关键点与 TRC 丢点审计报告

## 结论

- 这不是最近并行三角化改动把耳朵或头顶“删掉”了。当前仓库里，`trc` 输出哪些 marker，取决于 `pose_model` 对应的 skeleton。`make_trc()` 只会把上游选中的 `keypoints_names` 原样写进 TRC 表头。
- `Body_with_feet -> HALPE_26`：实际代码跑出来的 TRC 只有 **22** 个 marker，没有 `REye/LEye/REar/LEar/HeadTop`，但有 `Head`。
- `Whole_body_wrist -> COCO_133_WRIST`：实际代码跑出来的 TRC 有 **27** 个 marker，保留 `REye/LEye`，但仍然**没有耳朵**，也没有 `HeadTop`。
- `Whole_body -> COCO_133`：实际代码跑出来的 TRC 有 **131** 个 marker，`REye/LEye/REar/LEar` 都在，但仍然**没有 `HeadTop`**。
- 当前内置 skeleton 里没有任何 `HeadTop` 节点；如果 2D JSON 里额外带了头顶 triplet，三角化阶段也会因为 skeleton 没有这个 ID/名字而忽略它。

## 图示

![Pipeline](../figures/triangulation_keypoint_audit/fig01_pipeline.png)

![Presence](../figures/triangulation_keypoint_audit/fig02_presence_matrix.png)

![Headers](../figures/triangulation_keypoint_audit/fig03_trc_headers.png)

## 实际代码测试

审计脚本直接调用仓库现有的 `Pose2Sim.triangulation.triangulate_all()`，不是手工拼 TRC 字符串。输入 JSON 固定包含：

- `REye/LEye/REar/LEar` 四个额外 face triplet
- 一个额外的 `HeadTop` triplet（追加在 COCO_133 最大索引之后）
- 四台相机、两帧、合成标定与合成 2D 观测

运行结果汇总：

| row              |   REye |   LEye |   REar |   LEar |   Head |   HeadTop |
|:-----------------|-------:|-------:|-------:|-------:|-------:|----------:|
| 2D JSON input    |      1 |      1 |      1 |      1 |      0 |         1 |
| Body_with_feet   |      0 |      0 |      0 |      0 |      1 |         0 |
| Whole_body_wrist |      1 |      1 |      0 |      0 |      0 |         0 |
| Whole_body       |      1 |      1 |      1 |      1 |      0 |         0 |

| pose_model       | skeleton       |   trc_markers | REye   | LEye   | REar   | LEar   | Head   | HeadTop   |
|:-----------------|:---------------|--------------:|:-------|:-------|:-------|:-------|:-------|:----------|
| Body_with_feet   | HALPE_26       |            22 | no     | no     | no     | no     | yes    | no        |
| Whole_body_wrist | COCO_133_WRIST |            27 | yes    | yes    | no     | no     | no     | no        |
| Whole_body       | COCO_133       |           131 | yes    | yes    | yes    | yes    | no     | no        |

保存下来的运行证据：

- `data/triangulation_keypoint_audit/input_sample.json`
- `data/triangulation_keypoint_audit/Body_with_feet.trc`
- `data/triangulation_keypoint_audit/Whole_body_wrist.trc`
- `data/triangulation_keypoint_audit/Whole_body.trc`

## 代码链路证据

- `Pose2Sim/triangulation.py` 会先把 `pose_model` 映射到 skeleton，然后用 `RenderTree(model)` 生成 `keypoints_ids` 和 `keypoints_names`。
- 同文件里的 `make_trc()` 只根据传入的 `keypoints_names` 写 TRC 表头，不会再做一次“删耳朵/删头顶”的过滤。
- `Pose2Sim/poseEstimation.py` 里，`Whole_body_wrist` 和 `Whole_body` 都走 `COCO_133` RTMLib 检测模型；但 `triangulation.py` 里 `Whole_body_wrist` 会映射成 `COCO_133_WRIST` skeleton。这意味着 2D 可视化里能看到更多 face 点，不等于 TRC 会把这些点都写出来。

关键 `git blame`：

```text
df085524 (davidpagnon  2025-01-04 17:55:24 +0100 770)         if pose_model.upper() == 'BODY_WITH_FEET': pose_model = 'HALPE_26'
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 771)         elif pose_model.upper() == 'WHOLE_BODY_WRIST': pose_model = 'COCO_133_WRIST'
df085524 (davidpagnon  2025-01-04 17:55:24 +0100 772)         elif pose_model.upper() == 'WHOLE_BODY': pose_model = 'COCO_133'
df085524 (davidpagnon  2025-01-04 17:55:24 +0100 773)         elif pose_model.upper() == 'BODY': pose_model = 'COCO_17'
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 774)         elif pose_model.upper() == 'HAND': pose_model = 'HAND_21'
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 775)         elif pose_model.upper() == 'FACE': pose_model = 'FACE_106'
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 776)         elif pose_model.upper() == 'ANIMAL': pose_model = 'ANIMAL2D_17'
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 777)         else: pass
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 778)         model = eval(pose_model)
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 779)     except:
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 780)         try: # from Config.toml
b2fe4f7b (David PAGNON 2024-07-09 16:39:33 +0200 781)             model = DictImporter().import_(config_dict.get('pose').get(pose_model))
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 782)             if model.id == 'None':
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 783)                 model.id = None
b970c86d (davidpagnon  2023-11-01 15:56:37 +0100 784)         except:
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 785)             raise NameError('{pose_model} not found in skeletons.py nor in Config.toml')
2b8a721e (davidpagnon  2025-01-14 02:28:26 +0100 786)             
5bee7bf4 (davidpagnon  2024-01-04 17:20:06 +0100 787)     keypoints_ids = [node.id for _, _, node in RenderTree(model) if node.id!=None]
5bee7bf4 (davidpagnon  2024-01-04 17:20:06 +0100 788)     keypoints_names = [node.name for _, _, node in RenderTree(model) if node.id!=None]
```

最近提交历史（截断）：

```text
f4804ce Add batched pose inference and triangulation docs (#2)
2f27952 Automatic video discovery: no need for `vid_img_extension` anymore
88a072d Ensured backward compatibility by assigning default value in all configuration parameters
678397e Added average_likelihood_threshold to filter out bad ghost bounding boxes (tripod, for example)
c61ada9 Update angle calculation to avoid discontinuities and other inconsistencies
d429ad7 Edited signature of indices_of_first_last_non_nan_chunks for future change (not implemented yet)
fcec996 fixed excluded camera count + fixed poseEstimation with save_video = 'to_images' + clearer docstrings
4039c98 new sorting and cleaner triangulation
```

这说明核心 marker 映射行已经存在较久，不是这次并行三角化改动才引入的。

## 真实 trial 扫描

- 扫描结果为空：当前仓库里没有可直接审计的 `pose/pose-3d` 真实试次。

当前工作区没有用户真实 `pose/pose-3d` 试次可供逐文件核对，所以这份报告的运行证据全部来自可复现最小样例。它足够证明当前仓库代码的行为，但不能替代你本地那份真实 trial 的输入输出审计。

## 建议

- 如果你要在 TRC 里保留耳朵，不要用 `Body_with_feet`，也不要用 `Whole_body_wrist`；应改成 `pose_model = 'Whole_body'`，或者自定义 skeleton。
- 如果你要在 TRC 里保留 `HeadTop`，当前内置模型都不行。需要用 `pose.CUSTOM` 明确定义这个节点，并同步更新 OpenSim marker/下游流程。
- 如果你的 2D JSON 来自外部检测器，且里面已经有耳朵/头顶，但 Config 仍写成 `Body_with_feet` 或 `Whole_body_wrist`，那么这些额外 triplet 会在三角化阶段被忽略。这是配置与 skeleton 不匹配，不是最近三角化代码回归。
