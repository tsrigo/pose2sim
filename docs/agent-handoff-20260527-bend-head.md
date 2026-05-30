# Agent Handoff: 2026-05-27 PnVision 弯腰头部停留问题

## 当前用户请求

用户已补充真实数据，要求继续分析并修复：

- 样本：`data/PnVision/20260527_NewExtrinsics/yh856351_20260527_155254_身体前屈`
- 现象：这个版本增加 2D 平滑后，弯腰时经常出现头部还停留在上面。
- 交付物：压缩包里必须包含产物，包括 TRC，以及“2D 化之后的视频”，也就是视频里叠加了很多 2D pose 点的可视化视频。

本轮后来被用户中断，改为要求把所有必要信息记录到本 Markdown，交给另一个 agent 接手。

## 仓库和工作目录

工作目录：

```bash
/data/users/weikaihuang/projects/pose2sim
```

重要仓库规则来自用户提供的 `AGENTS.md`：

- bug/regression 必须尽量跑真实代码路径，不要只静态读代码。
- 修复前先复现，并保存失败证据。
- TRC / pose / triangulation 问题要追到具体阶段：输入 JSON、骨架映射、三角化、TRC 写出或下游转换。
- 修复后要用同一输入重跑并给 before/after 证据。
- 不要回滚用户或其他 agent 的改动。

## 已确认的真实样本状态

真实 trial 现在存在，包含：

- `Calib_board.toml`
- `cam01.avi` 到 `cam04.avi`
- `metadata.json`
- `record_diagnostics.json`
- `pose-3d/compute-pose.log`
- `pose-3d/yh856351_20260527_155254_身体前屈.trc`
- `assessment_visuals/*.jpg`

四路 AVI 基本信息：

- 全部可打开。
- 627 帧。
- 30 fps。
- 分辨率：`1240x1624`。

原始 assessment 信息：

- 动作：身体前屈。
- quality gate passed。
- 既有 TRC 有 568 帧。
- neutral window：503-512。
- selected bend frames：435-439。

## 已做代码改动

主要修改文件：

- `Pose2Sim/Utilities/avi_to_trc.py`
- `docs/hzvision-integration.md`

`avi_to_trc.py` 当前已经加入：

- `drop_low_average_pose` 配置项，默认 `false`。
- `VideoStats.low_average_pose_frames` 统计。
- `flush_pose_queue(..., drop_low_average_pose, ..., temporal_smoother=None)`：
  - 如果整个人平均 keypoint likelihood 低于 `average_likelihood_threshold_pose`，先计入 `low_average_pose_frames`。
  - 只有 `drop_low_average_pose = true` 时才把整帧 keypoints/scores 清空。
  - `drop_low_average_pose = false` 时保留各 keypoint 和各自 likelihood，让三角化阶段按单点 likelihood 过滤。
- `avi_to_trc` 路径接入了 2D temporal smoothing：
  - `_make_pose_temporal_smoother`
  - `pose_temporal_smoothing_config`
  - detection miss 写空帧时也经过 smoother。
- 日志会明确写：
  - `Low-average pose frames are dropped...`
  - 或 `Low-average pose frames are retained...`

`docs/hzvision-integration.md` 已加入推荐配置：

```toml
[pose]
temporal_smoothing = true
temporal_smoothing_method = "one_euro"
temporal_smoothing_min_cutoff = 1.0
temporal_smoothing_beta = 0.02
temporal_smoothing_d_cutoff = 1.0
temporal_smoothing_min_likelihood = 0.3
temporal_smoothing_max_gap = 5
temporal_smoothing_keypoints = "all"
drop_low_average_pose = false
```

并说明弯腰、遮挡、局部低分时不应因为整个人平均分低就清空整帧，否则 TRC 后续 `last_value` 填 gap 会让头部看起来停留在上面。

## 真实复现目录

已经基于真实数据创建两个对比 trial：

```text
data/repro_20260527_bend_head_real/
├── analysis_summary.csv
├── legacy_drop/
│   └── yh856351_20260527_155254_身体前屈/
│       ├── Config.toml
│       ├── calibration/Calib_board.toml
│       ├── logs.txt
│       ├── pose/
│       ├── pose-3d/yh856351_20260527_155254_身体前屈_0-563.trc
│       └── videos/ -> symlinks to real AVI files
└── fixed_retain/
    └── yh856351_20260527_155254_身体前屈/
        ├── Config.toml
        ├── calibration/Calib_board.toml
        ├── logs.txt
        ├── pose/
        ├── pose-3d/yh856351_20260527_155254_身体前屈_0-567.trc
        ├── videos/ -> symlinks to real AVI files
        └── videos-2d/fixed_pose2d_mosaic.mp4
```

两个 `Config.toml` 主要一致，差异是：

- legacy：`drop_low_average_pose = true`
- fixed：`drop_low_average_pose = false`

共同关键配置：

- `multi_person = false`
- `frame_rate = 30`
- `frame_range = 'all'`
- `pose_model = "Body_with_feet"`
- `mode = "performance"`
- `det_frequency = 4`
- `batch_size = 8`
- `output_format = "openpose"`
- `average_likelihood_threshold_pose = 0.5`
- `temporal_smoothing = true`
- OneEuro：`min_cutoff = 1.0`, `beta = 0.02`, `d_cutoff = 1.0`
- `temporal_smoothing_min_likelihood = 0.3`
- `temporal_smoothing_max_gap = 5`
- `temporal_smoothing_keypoints = "all"`
- triangulation threshold：`reproj_error_threshold_triangulation = 20`
- `likelihood_threshold_triangulation = 0.3`
- `min_cameras_for_triangulation = 3`
- interpolation：`linear`
- larger gaps fill：`last_value`
- pelvis rigid group enabled
- filtering disabled

## 已运行命令

legacy 跑法：

```bash
python -m Pose2Sim.Utilities.avi_to_trc \
  --trial-dir "data/repro_20260527_bend_head_real/legacy_drop/yh856351_20260527_155254_身体前屈" \
  --batch-size 8 \
  --overwrite-pose
```

fixed 跑法：

```bash
python -m Pose2Sim.Utilities.avi_to_trc \
  --trial-dir "data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈" \
  --batch-size 8 \
  --overwrite-pose
```

两次都走了真实 RTMPose/YOLOX + 四相机 triangulation 代码路径，使用 CUDA/ONNXRuntime。

## 复现结论

真实样本上已复现根因：

旧逻辑会把“整个人平均置信度低于 0.5”的帧整帧清空，即使其中仍有可用 keypoint。弯腰尾段因为头部/局部点置信度低，这会制造整帧 2D 空洞。三角化之后再用 `last_value` 填较大 gap，就会表现为头部停在上面。

fixed 逻辑保留这些低平均分姿态帧，把过滤交给后续 per-keypoint likelihood 和 triangulation。

## legacy vs fixed 关键数字

legacy 日志：

```text
Low-average pose frames are dropped when mean keypoint likelihood is below 0.5.
cam01.avi: processed 627 frames, dropped 70, detection misses 59, low-average poses 11
cam02.avi: processed 627 frames, dropped 63, detection misses 59, low-average poses 4
cam03.avi: processed 627 frames, dropped 33, detection misses 28, low-average poses 5
cam04.avi: processed 627 frames, dropped 0, detection misses 0, low-average poses 0
```

fixed 日志：

```text
Low-average pose frames are retained; per-keypoint likelihoods will be handled by triangulation.
cam01.avi: processed 627 frames, dropped 59, detection misses 59, low-average poses 11
cam02.avi: processed 627 frames, dropped 59, detection misses 59, low-average poses 4
cam03.avi: processed 627 frames, dropped 28, detection misses 28, low-average poses 5
cam04.avi: processed 627 frames, dropped 0, detection misses 0, low-average poses 0
```

也就是说 fixed 保留了真实样本里被 legacy 清空的 20 个低平均分 2D pose 帧：

- cam01：557-567，共 11 帧。
- cam02：564-567，共 4 帧。
- cam03：591-595，共 5 帧。

`analysis_summary.csv` 内容摘要：

```text
legacy_drop cam01 empty 557-626, lowavg_retained 0
legacy_drop cam02 empty 564-626, lowavg_retained 0
legacy_drop cam03 empty 591-622,624, lowavg_retained 0
legacy_drop cam04 empty none, lowavg_retained 0

fixed_retain cam01 empty 568-626, lowavg_retained 557-567
fixed_retain cam02 empty 568-626, lowavg_retained 564-567
fixed_retain cam03 empty 596-622,624, lowavg_retained 591-595
fixed_retain cam04 empty none, lowavg_retained 0
```

TRC 输出：

```text
legacy: data/repro_20260527_bend_head_real/legacy_drop/yh856351_20260527_155254_身体前屈/pose-3d/yh856351_20260527_155254_身体前屈_0-563.trc
fixed:  data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/pose-3d/yh856351_20260527_155254_身体前屈_0-567.trc
```

Triangulation 日志：

```text
legacy: Mean reprojection error for all points on frames 0 to 564 is 9.3 px, roughly 24.2 mm.
legacy: trial trimmed between frames [0, 564].

fixed:  Mean reprojection error for all points on frames 0 to 568 is 9.3 px, roughly 24.2 mm.
fixed:  trial trimmed between frames [0, 568].
```

头部相关 reprojection error 基本不变：

```text
legacy: Head 8.8 px, Neck 12.9 px, Nose 14.0 px
fixed:  Head 8.8 px, Neck 12.9 px, Nose 14.0 px
```

解释：本修复的目标不是降低整体 project/reprojection error，而是避免可用 2D pose 被整帧清空，减少由 gap filling 造成的“头停住”。fixed TRC 比 legacy 多保留到第 567 帧。

## 仍需注意的限制

fixed 后尾段仍然存在检测 miss 或低 keypoint likelihood：

- cam01 fixed 仍从 568-626 为空。
- cam02 fixed 仍从 568-626 为空。
- cam03 fixed 仍有 596-622 和 624 为空。
- cam04 一直有 pose。

所以 fixed 解决的是“低平均分但仍有 pose 时被整帧清空”的 bug；它不能凭空恢复真正没有 detector/person 的尾段。尾段如果仍出现 head/neck last-value，下一步应继续看 detection miss、关键点逐点 likelihood 或是否需要调整更高层 tracking/detector 策略。

## 已生成 2D 可视化视频

已经生成 fixed 版本四路相机拼接的 2D 点位叠加视频：

```text
data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/videos-2d/fixed_pose2d_mosaic.mp4
```

视频信息：

- 627 帧。
- 30 fps。
- 画面大小：`868x1136`。
- 四路相机 2x2 拼接。
- 在原始视频上叠加 HALPE_26/OpenPose-style 2D 骨架点和连线。
- 文件大小约 15 MB。

生成脚本是一次性 inline Python，没有保存成仓库脚本。若要重建，可从 shell history 或本轮记录恢复；更简单是直接使用已生成 mp4。

## 接手后的建议任务

1. 检查当前 git diff，确认 `Pose2Sim/Utilities/avi_to_trc.py` 的改动还在。
2. 检查 `analysis_summary.csv`、legacy/fixed logs 和 TRC 是否满足交付证据。
3. 生成一个简短 `README.md` 或 `HANDOFF_SUMMARY.md` 放在 `data/repro_20260527_bend_head_real/`，方便压缩包内阅读。
4. 打包 zip，至少包含：
   - `Pose2Sim/Utilities/avi_to_trc.py`
   - `docs/hzvision-integration.md`
   - 本 handoff 文件
   - `data/repro_20260527_bend_head_real/analysis_summary.csv`
   - legacy/fixed 两个 `Config.toml`
   - legacy/fixed 两个 `logs.txt`
   - legacy/fixed 两个 TRC
   - fixed 的 `videos-2d/fixed_pose2d_mosaic.mp4`
5. 用 `zip -T` 校验压缩包。

建议压缩包路径：

```text
deliverables/pnvision_20260527_bend_head_real_fix_20260527.zip
```

可用打包命令示例：

```bash
mkdir -p deliverables
zip -r deliverables/pnvision_20260527_bend_head_real_fix_20260527.zip \
  Pose2Sim/Utilities/avi_to_trc.py \
  docs/hzvision-integration.md \
  docs/agent-handoff-20260527-bend-head.md \
  data/repro_20260527_bend_head_real/analysis_summary.csv \
  data/repro_20260527_bend_head_real/legacy_drop/yh856351_20260527_155254_身体前屈/Config.toml \
  data/repro_20260527_bend_head_real/legacy_drop/yh856351_20260527_155254_身体前屈/logs.txt \
  data/repro_20260527_bend_head_real/legacy_drop/yh856351_20260527_155254_身体前屈/pose-3d/yh856351_20260527_155254_身体前屈_0-563.trc \
  data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/Config.toml \
  data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/logs.txt \
  data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/pose-3d/yh856351_20260527_155254_身体前屈_0-567.trc \
  data/repro_20260527_bend_head_real/fixed_retain/yh856351_20260527_155254_身体前屈/videos-2d/fixed_pose2d_mosaic.mp4

zip -T deliverables/pnvision_20260527_bend_head_real_fix_20260527.zip
```

## 有用命令

查看当前相关 diff：

```bash
git diff -- Pose2Sim/Utilities/avi_to_trc.py docs/hzvision-integration.md
```

查看 evidence 文件：

```bash
find data/repro_20260527_bend_head_real -maxdepth 4 -type f | sort
```

查看关键日志：

```bash
rg -n "Low-average|processed|Mean reprojection|trial trimmed" \
  data/repro_20260527_bend_head_real/*/yh856351_20260527_155254_身体前屈/logs.txt
```

原始 `compute-pose.log` 里有 NUL 字节，`rg` 会提示 binary file matches。需要解析时建议用 Python `errors="ignore"` 或 `strings`。

## 交付时建议说明

最终对用户可以这样概括：

- 已在真实补充数据上复现问题。
- 根因是旧逻辑在弯腰尾段把 20 个低平均分但仍有 2D pose 的帧整帧清空。
- 修复后这些帧被保留，三角化按逐点 likelihood 处理；fixed TRC 从 0-563 延长到 0-567。
- 整体 reprojection error 仍为 9.3 px，说明这次修的是 head freeze/gap 的数据保留问题，不是降低 project error 的改动。
- zip 中已/应包含 fixed TRC、legacy/fixed 对比证据、日志、配置和 2D 点位视频。
