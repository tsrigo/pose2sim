# 单人场景下是否需要 personAssociation

## 结论先说

如果满足下面这个严格条件：

- 每一帧、每一路相机的输入 JSON 里都只有 1 个目标人

那么单人场景下可以跳过 `personAssociation`，直接从 `pose/*.json` 做三角化，最终 TRC 不会因此改变。

但在实际工程里，“物理上只有一个人”通常不等于“原始 pose JSON 里只有一个人”。只要有路人、误检、反光、多框，或者某一路相机经常检测出额外 person，就仍然建议保留 `personAssociation`。

## 流程图

![单人场景下 personAssociation 的判断流程](../figures/single-person-personassociation-flow.png)

## 代码上为什么可以跳过

Pose2Sim 的三角化阶段会按下面这个顺序寻找输入：

1. `pose-associated`
2. `pose-sync`
3. `pose`

这意味着：

- 如果你已经有 `pose-associated/*.json`，三角化会优先用它
- 如果没有 `pose-associated`，但 `pose/*.json` 本身就已经是“每帧每相机一个目标人”，三角化仍然可以直接工作

`personAssociation` 在单人模式下做的事，本质上是：

- 枚举各相机中可能的 person 组合
- 用 `tracked_keypoint` 的重投影误差挑出最像“同一个人”的组合
- 重写出只保留目标人的 `pose-associated/*.json`

所以它不是三角化的硬前置条件，而是“把原始多 person JSON 清洗成单目标 JSON”的一道筛选步骤。

## 本地实跑研究

本次研究基于 `Pose2Sim/Demo_SinglePerson` 做了 12 帧对照实验，并额外做了一组人为注入干扰人的实验。

实验路径有 4 组：

- `clean-skip`：原始 `pose/*.json` 直接三角化
- `clean-with`：先跑 `personAssociation`，再三角化
- `oracle-single-direct`：把 `clean-with` 生成的单人 `pose-associated/*.json` 直接当成 `pose/*.json`，跳过 `personAssociation` 再三角化
- `distractor-skip / distractor-with`：在部分帧中人为注入错误的额外 person，再分别测试跳过和保留 `personAssociation`

## 关键发现

### 1. 单人 demo 的原始 JSON 并不总是只有 1 个人

这次本地实跑里，`cam01_json` 的 `12/12` 帧都出现了 `>1 person`，其余 3 路相机是 `1 person`。

这说明一个关键事实：

- 单人实验场景，并不自动意味着 `poseEstimation` 输出就是单人 JSON

## 结果图

![单人场景 personAssociation 实测结果](../figures/single-person-personassociation-results.png)

## 结果表

下表用 `clean-with` 作为参考 TRC，比较其他条件的 3D 差异：

| 条件 | 含义 | 相对参考 TRC 的 RMS 差异 |
| --- | --- | ---: |
| `clean-skip` | 原始 JSON 直接三角化 | 23.9 mm |
| `oracle-single-direct` | 已经变成单人 JSON 后，跳过 `personAssociation` 直接三角化 | 0.0 mm |
| `distractor-skip` | 注入错误 person 后，跳过 `personAssociation` | 213.4 mm |
| `distractor-with` | 注入错误 person 后，保留 `personAssociation` | 0.0 mm |

这组结果说明：

- 一旦输入 JSON 已经被约束成“每帧每相机只有一个目标人”，跳过 `personAssociation` 不会改变结果
- 但只要原始 JSON 里混入额外 person，跳过 `personAssociation` 就可能把 TRC 拉偏很多

## 实际建议

下面这条判断最稳：

- 如果你能确认 `pose/*.json` 已经是单人且目标正确：可以跳过 `personAssociation`
- 如果只是“场景里大概率只有一个人”，但检测结果里可能混入其他 person：建议保留 `personAssociation`
- 如果你走的是 `avi_to_trc` 这条单人直通路径：在干净单人场景下通常可以不单独跑 `personAssociation`

## 适合发给老师的一句话

单人场景下，`personAssociation` 不是三角化的硬前置步骤；如果每帧每相机的 JSON 已经只剩目标人，可以直接做 triangulation。  
但实际单人实验里原始 pose JSON 仍可能混入额外 person，这时保留 `personAssociation` 更稳，本地实跑里跳过它会带来约 `23.9 mm` 到 `213.4 mm` 的 3D 偏差。
