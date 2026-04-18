# 本地 Smoke Test 证据

## 目标

验证新的 `avi_to_trc` 路径可以端到端跑通：

1. 读取 Pose2Sim 风格 trial 中的 `videos/*.avi`
2. 以批量 RTMPose crop 推理导出 OpenPose JSON
3. 三角化生成标准 `.trc`

## 本地测试设置

此次 smoke run 使用的临时 session：

- Session 根目录：`/tmp/avi-to-trc-test-ATAssR/Session`
- Trial 根目录：`/tmp/avi-to-trc-test-ATAssR/Session/Trial_1`

源数据：

- 视频来自 `Pose2Sim/Demo_SinglePerson/videos/*.mp4`
- 文件名被改成了 `*.avi`
- 标定从 `Calib.qca.txt` 转成了 `Calib_qualisys.toml`

本次 smoke run 使用的配置覆盖：

- `frame_range = [0, 8]`
- `mode = 'lightweight'`
- `det_frequency = 1`
- `device = 'cpu'`
- `backend = 'openvino'`
- `parallel_triangulation = false`
- `make_c3d = false`
- `min_chunk_size = 1`

这里额外把 `min_chunk_size` 设为 `1`，只是因为这次 smoke test 只有 `8` 帧。默认配置要求至少 `10` 个连续有效帧，否则会把这么短的 trial 过滤掉。

## 运行命令

```bash
python Pose2Sim/Utilities/avi_to_trc.py \
  --trial-dir /tmp/avi-to-trc-test-ATAssR/Session/Trial_1 \
  --batch-size 4 \
  --overwrite-pose
```

## 观测到的输出

生成的 pose JSON 目录：

```text
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam01_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam02_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam03_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam04_json
```

生成的 JSON 数量：

```text
cam01_json 8
cam02_json 8
cam03_json 8
cam04_json 8
```

生成的 TRC：

- `/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc`

TRC 头部摘录：

```text
PathFileType	4	(X/Y/Z)	Trial_1_0-7.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
60	60	8	22	m	60	0	8
Frame#	Time	Hip			RHip			RKnee			RAnkle			RBigToe			RSmallToe			RHeel			LHip			LKnee			LAnkle			LBigToe			LSmallToe			LHeel			Neck			Head			Nose			RShoulder			RElbow			RWrist			LShoulder			LElbow			LWrist
```

TRC 结构检查：

```text
frames 8
markers 22
frame_range 0 7
time_range 0.0 0.1166666666666666
first_markers ['Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe']
```

## 运行时现象摘录

CLI 输出的每路相机摘要：

```text
cam01.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam02.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam03.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam04.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
```

现有三角化流程输出的摘要：

```text
--> Mean reprojection error for all points on frames 0 to 8 is 11.0 px, which roughly corresponds to 20.0 mm.
3D coordinates are stored at /tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc.
```

## 结论

这条新路径已经在真实本地运行中端到端跑通：

- `*.avi` 输入被正确接受
- 4 路相机的批量 pose JSON 导出成功
- 三角化无须改格式即可直接消费这些 JSON
- 最终生成了标准 3D `.trc`
