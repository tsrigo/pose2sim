# Summary

This PR adds a new `avi_to_trc` CLI that runs batched RTMLib inference on
Pose2Sim-style multi-camera `*.avi` trials and then triangulates the exported
OpenPose JSONs into a standard 3D `.trc`.

# What Changed

- added `Pose2Sim/Utilities/avi_to_trc.py`
- registered the new `avi_to_trc` console script in `pyproject.toml`
- added end-user documentation in `docs/avi_to_trc.md`
- added milestone-local validation evidence in `context/avi_to_trc_cli/*`

# Behavior

- accepts a trial directory containing `videos/*.avi`
- resolves config from trial/session `Config.toml`
- requires an existing calibration `.toml`
- exports OpenPose JSONs under `pose/*_json`
- reuses existing Pose2Sim triangulation to write the final 3D `.trc`
- single-person only in v1
- rejects one-stage RTMLib models in v1
- batches pose crop inference, while detection remains frame-wise

# Validation

Static checks:

- `python -m compileall Pose2Sim/Utilities/avi_to_trc.py`
- `python Pose2Sim/Utilities/avi_to_trc.py --help`

End-to-end local smoke test:

- temp session root: `/tmp/avi-to-trc-test-ATAssR/Session`
- command:

```bash
python Pose2Sim/Utilities/avi_to_trc.py \
  --trial-dir /tmp/avi-to-trc-test-ATAssR/Session/Trial_1 \
  --batch-size 4 \
  --overwrite-pose
```

- outputs:
  - 4 pose JSON folders with 8 frames each
  - TRC at `/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc`

Full local evidence is recorded in `context/avi_to_trc_cli/smoke-test.md`.
