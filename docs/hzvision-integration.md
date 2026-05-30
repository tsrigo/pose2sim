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
