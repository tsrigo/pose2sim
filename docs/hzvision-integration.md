# HZVision 集成与分支维护

`pose2sim-hzvision` 是给 HZVision 调用的 Pose2Sim 长期维护分支。该分支保留 Pose2Sim 原始提交历史，并在此基础上维护 `avi_to_trc` 入口和 PnVision/HZVision 所需的稳定性改动。

## 分支定位

- `master` 或其他 HZVision 主程序分支不直接承载 Pose2Sim 源码。
- `pose2sim-hzvision` 承载完整 Pose2Sim 源码、`avi_to_trc` CLI 和 HZVision 相关算法改动。
- HZVision 主程序通过安装该分支或指定源码路径调用 Pose2Sim，不再手工替换虚拟环境中的 `pose2sim` 文件夹。
- 本分支不提交真实业务数据、临时实验输出、交付 zip 或下载中的视频文件。

## 调用方式

推荐把该分支安装到 HZVision 的运行环境后调用命令行入口：

```bash
avi_to_trc --trial-dir /path/to/Session/Trial_1 --batch-size 16 --overwrite-pose
```

也可以从 Python 中调用：

```python
from Pose2Sim.Utilities.avi_to_trc import avi_to_trc

avi_to_trc(
    trial_dir="/path/to/Session/Trial_1",
    batch_size=16,
    overwrite_pose=True,
)
```

`trial_dir` 需要满足 Pose2Sim trial 结构：trial 下包含 `videos/*.avi`，session 下包含 `calibration/*.toml`，配置文件可以放在 session 或 trial 下。

## 2D 检测点时序去抖

如果 RTMPose 在人静止时脚点、髋点仍然逐帧抖动，应优先在 `pose` 阶段对 2D keypoint 做轻量时序平滑，再进入三角化。

推荐配置：

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

这个开关在 RTMPose 推理、pose-level NMS 和 person tracking 之后执行，在写 OpenPose JSON 和绘制检测视频之前执行。因此 `pose/*.json`、检测可视化和后续三角化会使用同一套稳定后的 2D 点。默认关闭；不开启时现有输出不变。

`drop_low_average_pose = false` 用于弯腰、遮挡、手脚出画等局部低分场景：不要因为整个人所有 keypoint 的平均分低于 `average_likelihood_threshold_pose` 就清空整帧。各点原始 likelihood 会保留下来，后续三角化仍按 `likelihood_threshold_triangulation` 逐点过滤。否则弯腰时头部短暂低分可能让整帧 2D pose 变空，TRC 再用 last-value 填 gap，看起来就像头还停在上面。

## 站立片段的 3D 稳定配置

如果 trial 本身是站立或近似静止动作，只做 2D 平滑和髋部刚体保护还可能留下 3D 三角化层面的 Neck、Hip 单点跳动。四相机数据里尤其要避免单个 marker 退化成两相机解，因为两相机 reprojection error 可以很低，但 3D 点会在两组相机对之间切换。

这类站立片段建议先提高三角化最低相机数，并把三角化 reprojection 阈值收紧到 10px 左右，再做 3D 滤波。阈值过松时，四相机解即使已经被某一路坏点拉高到十几像素，也不会触发相机剔除；在 `yh644406_20260527_114626_静态站立` 的 517-526 帧窗口里，20px 阈值会保留所有相机并得到 8.8px 平均误差，10px 阈值会剔除少量坏观测并降到 7.1px。

```toml
[triangulation]
reproj_error_threshold_triangulation = 10
min_cameras_for_triangulation = 3
```

然后显式启用 Pose2Sim 现有的 3D outlier rejection 和 Butterworth 滤波，并使用偏保守的低截止频率：

```toml
[filtering]
reject_outliers = true
filter = true
type = "butterworth"

[filtering.butterworth]
cut_off_frequency = 3
order = 4
```

该配置适合站立稳定性优先的片段。对于跑跳、快速摆臂等真实快速运动，应提高截止频率或使用项目原有默认值，避免过度平滑真实动作。

## 刚体稳定配置

本分支增加了两个可选稳定步骤，默认不启用；只有在 `Config.toml` 中配置 `rigid_marker_groups` 时才会生效。

三角化阶段可配置：

```toml
[triangulation]
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"] }
]
rigid_group_min_markers = 3
rigid_group_smoothing_window = 5
rigid_group_reproj_error_threshold = 15
rigid_group_blend = 0.7
rigid_group_max_correction_m = 0.05
rigid_group_max_pairwise_change_ratio = 0.15
rigid_group_fill_missing = false
rigid_group_max_cams_to_exclude = 1
rigid_group_max_nfev = 30
```

滤波阶段可配置：

```toml
[filtering]
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"] }
]
rigid_group_min_markers = 3
rigid_group_smoothing_window = 31
rigid_group_blend = 0.7
rigid_group_max_correction_m = 0.05
rigid_group_max_pairwise_change_ratio = 0.15
rigid_group_max_correction_step_m = 0.015
rigid_group_fill_missing = false
rigid_group_repair_pairwise_spikes = true
rigid_group_spike_window = 31
rigid_group_spike_max_gap = 5
rigid_group_spike_pairwise_change_ratio = 0.25
rigid_group_spike_min_abs_m = 0.03
rigid_group_spike_min_bad_pairs = 2
```

这些配置用于降低同一刚体局部 marker 的相对抖动，例如 PnVision 腰部 `Hip/RHip/LHip`。三角化阶段和滤波阶段现在默认都是近似刚体：用刚体拟合结果稳定轨迹，但会混回 baseline，并限制单点修正量、组内距离变化和修正额外引入的逐帧位移；对已经明显偏离局部模板的帧，只接受能让组内距离更接近模板的修正，避免髋部比例被压缩或在开关修正时出现突变。`rigid_group_repair_pairwise_spikes` 只在显式开启时处理短时组内距离尖峰：它会找出最可能出错的局部 marker，并只对短片段做相邻有效帧插值。如果没有配置，现有 Pose2Sim 行为保持不变。

## 生产级稳定增强（弯腰 / 转身 / 行走 / 遮挡）

在基础刚体组之上，本分支针对真实 PnVision 试验中观察到的伪影增加了若干**默认开启**的护栏（仅在配置了对应刚体组时才参与计算；不配置刚体组则现有行为不变）。这些选项写在 `rigid_marker_groups` 条目里（短名会自动加前缀 `rigid_group_*`），实现都在 `Pose2Sim/triangulation.py`。

```toml
[triangulation]
rigid_marker_groups = [
  { name = "pelvis", markers = ["Hip", "RHip", "LHip"],
    blend = 1.0, fill_missing = true,
    reproj_error_threshold = 45, smoothing_window = 9,
    smoothing_method = "robust", orientation_window = 15, orientation_tol_deg = 35,
    chirality_guard = true, chirality_hold = true,
    twist_guard = true, max_twist_deg = 50 },
  { name = "head",   markers = ["Head","Nose","REye","LEye","REar","LEar"],
    blend = 1.0, fill_missing = true,
    reproj_error_threshold = 25, smoothing_window = 9 }
]

# 非物理肢体护栏（全局，默认开启）
reject_nonphysical_limbs = true
limb_length_max_ratio = 1.8
limb_length_min_excess_m = 0.15
```

各护栏要解决的问题：

- **手性护栏 `chirality_guard` + `chirality_hold`**：行走时骨盆侧向轴沿相机深度方向被透视压缩（foreshortening）+ 背向相机，刚体拟合会以几乎为零的 2D 代价 180° 翻转，导致 RHip/LHip 左右互换。护栏以独立三角化（逐 marker，不可能换边）为基准方向，翻转帧拒绝刚体结果；基准为 NaN 时用时间连续性（相邻 30fps 帧 yaw 不可能突变 180°）。`chirality_hold` 在拒绝帧不回退到退化的独立解，而是把上一可信朝向以当前质心重定位后保持，避免骨盆宽度塌缩。
- **鲁棒 SO(3) 朝向平滑 `smoothing_method = "robust"`**：foreshortening 下 yaw 病态，逐分量中值无法剔除轴角向量上的非线性翻转尖峰。改为在 SO(3) 上取测地中值点（geodesic medoid），保留渐变真实运动、剔除尖峰。
- **解剖学扭转钳制 `twist_guard`**：刚体骨盆与躯干无耦合，会"像光滑的杆"过度旋转到非解剖角度。以肩线（RShoulder/LShoulder）为参考、绕竖直轴把骨盆相对躯干扭转钳制在 `max_twist_deg`（默认 50°，真实躯干旋转约 45°，故只在伪影帧触发）。
- **Procrustes / Kabsch 刚体模板**：旧模板对居中后位置逐轴取中值，身体转动时会把形状压缩（不同朝向的 marker 互相抵消）。改为先用 Kabsch 把每个稳定模板帧旋到公共参考帧再取中值，无论是否转动都保住真实尺寸（yh644406 静态骨盆 14.5cm→22.4cm）。
- **非物理肢体护栏 `reject_nonphysical_limbs`**：两相机分歧时独立三角化的手臂/肩膀可能炸到 1–2 m。沿骨骼树自根向叶，把同时超过 `max_ratio`×试验中值长度且绝对超出 `min_excess_m` 的骨段，沿当前方向把远端子节点拉回中值长度。`+0.15m` 绝对下限保证脚部正常抖动不被误伤；刚体组 marker 受保护不被移动。

验证脚本：`scripts/verify_pelvis_jitter.py`（骨盆抖动/翻转/扭转指标，按 marker **名字** 解析，注意 TRC 中 RShoulder=21、LShoulder=24）、`scripts/verify_limb_guard.py`（肢体长度护栏前后对比）、`scripts/sweep_artifacts.py`（多试验骨段漂移 / L-R 互换 / 瞬移 / 非解剖扭转扫描）、`scripts/batch_validate.py` + `scripts/render_skeleton.py`（批量 + 骨架可视化 PNG 复核）、`scripts/audit_2d.py`（区分 2D 检测层 vs 三角化层伪影）。

## 同步官方 Pose2Sim

建议保留两个远端：

```bash
git remote add upstream <official-pose2sim-url>
git remote add hzvision http://platgit.ng.noitom.com.cn/scm/pnvis/hzvision.git
```

同步流程：

```bash
git fetch upstream
git switch pose2sim-hzvision
git merge upstream/main
```

如果官方更新与 HZVision 改动发生冲突，在 `pose2sim-hzvision` 上解决冲突并重新跑 `avi_to_trc` 和刚体稳定性验证。不要把账号、密码或真实数据写入 Git 提交。
