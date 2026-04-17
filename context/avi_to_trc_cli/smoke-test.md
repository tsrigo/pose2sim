# Local Smoke Test Evidence

## Goal

Verify the new `avi_to_trc` path end to end:

1. read a Pose2Sim-style trial with `videos/*.avi`
2. export OpenPose JSONs with batched RTMPose crop inference
3. triangulate into a standard `.trc`

## Local Test Setup

Temporary session used for the smoke run:

- Session root: `/tmp/avi-to-trc-test-ATAssR/Session`
- Trial root: `/tmp/avi-to-trc-test-ATAssR/Session/Trial_1`

Source assets:

- videos copied from `Pose2Sim/Demo_SinglePerson/videos/*.mp4`
- renamed to `*.avi`
- calibration converted from `Calib.qca.txt` to `Calib_qualisys.toml`

Config overrides used for the smoke run:

- `frame_range = [0, 8]`
- `mode = 'lightweight'`
- `det_frequency = 1`
- `device = 'cpu'`
- `backend = 'openvino'`
- `parallel_triangulation = false`
- `make_c3d = false`
- `min_chunk_size = 1`

The `min_chunk_size` override was only needed because the smoke test uses 8
frames. The default config expects at least 10 valid consecutive frames and
would reject such a short trial.

## Command

```bash
python Pose2Sim/Utilities/avi_to_trc.py \
  --trial-dir /tmp/avi-to-trc-test-ATAssR/Session/Trial_1 \
  --batch-size 4 \
  --overwrite-pose
```

## Observed Outputs

Generated pose JSON directories:

```text
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam01_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam02_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam03_json
/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose/cam04_json
```

Generated JSON counts:

```text
cam01_json 8
cam02_json 8
cam03_json 8
cam04_json 8
```

Generated TRC:

- `/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc`

TRC header excerpt:

```text
PathFileType	4	(X/Y/Z)	Trial_1_0-7.trc
DataRate	CameraRate	NumFrames	NumMarkers	Units	OrigDataRate	OrigDataStartFrame	OrigNumFrames
60	60	8	22	m	60	0	8
Frame#	Time	Hip			RHip			RKnee			RAnkle			RBigToe			RSmallToe			RHeel			LHip			LKnee			LAnkle			LBigToe			LSmallToe			LHeel			Neck			Head			Nose			RShoulder			RElbow			RWrist			LShoulder			LElbow			LWrist
```

TRC structural check:

```text
frames 8
markers 22
frame_range 0 7
time_range 0.0 0.1166666666666666
first_markers ['Hip', 'RHip', 'RKnee', 'RAnkle', 'RBigToe']
```

## Selected Runtime Observations

Per-camera summary reported by the CLI:

```text
cam01.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam02.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam03.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
cam04.avi: processed 8 frames, dropped 0, detection misses 0, pose batches 2.
```

Triangulation summary reported by the existing pipeline:

```text
--> Mean reprojection error for all points on frames 0 to 8 is 11.0 px, which roughly corresponds to 20.0 mm.
3D coordinates are stored at /tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc.
```

## Conclusion

The new path completed end to end on a real local run:

- `*.avi` input accepted
- batched pose JSON export completed for all 4 cameras
- triangulation consumed the exported JSONs without format changes
- a standard 3D `.trc` was produced
