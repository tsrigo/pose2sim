# 摘要

本 PR 新增了一个 `avi_to_trc` CLI，用于在 Pose2Sim 风格的多机位 `*.avi` trial 上运行批量 RTMLib 推理，并把导出的 OpenPose JSON 继续三角化成标准 3D `.trc`。

# 变更内容

- 新增 `Pose2Sim/Utilities/avi_to_trc.py`
- 在 `pyproject.toml` 中注册 `avi_to_trc` console script
- 新增面向用户的文档 `docs/avi_to_trc.md`
- 新增里程碑本地验证材料 `context/avi_to_trc_cli/*`

# 行为说明

- 接受一个包含 `videos/*.avi` 的 trial 目录
- 从 trial / session 的 `Config.toml` 解析配置
- 要求 session 下已有 `.toml` 标定文件
- 在 `pose/*_json` 下导出 OpenPose JSON
- 复用现有 Pose2Sim 三角化生成最终 3D `.trc`
- v1 只支持单人场景
- v1 不支持 one-stage RTMLib 模型
- pose crop 推理是批量的，但 detection 仍然按帧执行

# 验证

静态检查：

- `python -m compileall Pose2Sim/Utilities/avi_to_trc.py`
- `python Pose2Sim/Utilities/avi_to_trc.py --help`

端到端本地 smoke test：

- 临时 session 根目录：`/tmp/avi-to-trc-test-ATAssR/Session`
- 运行命令：

```bash
python Pose2Sim/Utilities/avi_to_trc.py \
  --trial-dir /tmp/avi-to-trc-test-ATAssR/Session/Trial_1 \
  --batch-size 4 \
  --overwrite-pose
```

- 输出结果：
  - 4 个 pose JSON 目录，每个目录 8 帧
  - 生成 TRC：`/tmp/avi-to-trc-test-ATAssR/Session/Trial_1/pose-3d/Trial_1_0-7.trc`

完整本地证据见 `context/avi_to_trc_cli/smoke-test.md`。
