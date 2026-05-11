# 文档

- [批量姿态推理与并行三角化](batch-pose-and-parallel-triangulation.md)：说明 `batch_size`、`parallel_workers_pose`、`parallel_triangulation` 的作用、限制和推荐用法。
- [AVI 到 TRC 命令行工具](avi_to_trc.md)：说明如何从 `videos/*.avi` 直接运行 RTMLib 推理并生成最终的 `.trc` 文件。
- [单人场景下是否需要 personAssociation](single-person-personassociation.md)：基于实际代码运行，说明什么情况下可以跳过 `personAssociation`，什么情况下不建议跳过。
- [HZVision 集成与分支维护](hzvision-integration.md)：说明 `pose2sim-hzvision` 分支定位、调用方式、刚体稳定配置和同步官方 Pose2Sim 的维护流程。
