#!/usr/bin/env python

from __future__ import annotations

import json
import subprocess
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "trc_head_face_fix"
FIG_DIR = ROOT / "figures" / "trc_head_face_fix"
DOC_PATH = ROOT / "docs" / "trc-head-face-fix.md"

MODELS = ["COCO", "BODY_25", "BODY_25B", "BODY_135"]
INTERESTING = ["REye", "LEye", "REar", "LEar", "Head", "HeadTop"]

OPENPOSE_OUTPUT_DOC_URL = "https://github.com/CMU-Perceptual-Computing-Lab/openpose/blob/master/doc/02_output.md"
OPENPOSE_BODY25B_BODY135_URL = "https://huggingface.co/camenduru/openpose/blob/5e17f6ad43ab415a0114537541a8d37d2503424f/src/openpose/pose/poseParameters.cpp"


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DOC_PATH.parent.mkdir(parents=True, exist_ok=True)


def load_summary(stage: str) -> dict[str, dict[str, object]]:
    summary_path = DATA_DIR / stage / "summary.json"
    items = json.loads(summary_path.read_text(encoding="utf-8"))
    return {item["pose_model"]: item for item in items}


def git_diff_excerpt() -> str:
    completed = subprocess.run(
        ["git", "diff", "--", "Pose2Sim/skeletons.py"],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
    )
    return completed.stdout.strip()


def wrap_markers(markers: list[str], width: int = 52) -> str:
    return textwrap.fill(", ".join(markers), width=width)


def build_presence_dataframe(before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        for stage, payload in [("before", before[model]), ("after", after[model])]:
            rows.append(
                {
                    "row": f"{model} {stage}",
                    **{marker: int(payload.get(f"has_{marker}", False)) for marker in INTERESTING},
                }
            )
    return pd.DataFrame(rows).set_index("row")


def build_count_dataframe(before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]) -> pd.DataFrame:
    rows = []
    for model in MODELS:
        rows.append(
            {
                "pose_model": model,
                "before": before[model]["marker_count"],
                "after": after[model]["marker_count"],
            }
        )
    return pd.DataFrame(rows)


def make_presence_figure(presence_df: pd.DataFrame) -> None:
    values = presence_df.values
    fig, ax = plt.subplots(figsize=(10.5, 5.0))
    ax.imshow(values, cmap="Greens", vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(presence_df.columns)), labels=presence_df.columns)
    ax.set_yticks(np.arange(len(presence_df.index)), labels=presence_df.index)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            ax.text(col_idx, row_idx, "yes" if values[row_idx, col_idx] else "no", ha="center", va="center", fontsize=9)

    ax.set_title("TRC Marker Presence Before vs After Fix")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_presence_before_after.png", dpi=200)
    plt.close(fig)


def make_count_figure(count_df: pd.DataFrame) -> None:
    x_pos = np.arange(len(count_df))
    width = 0.35
    fig, ax = plt.subplots(figsize=(9.0, 4.0))
    ax.bar(x_pos - width / 2, count_df["before"], width, label="Before")
    ax.bar(x_pos + width / 2, count_df["after"], width, label="After")
    ax.set_xticks(x_pos, labels=count_df["pose_model"])
    ax.set_ylabel("TRC marker count")
    ax.set_title("TRC Marker Count Change")
    ax.legend()
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_marker_count.png", dpi=200)
    plt.close(fig)


def make_header_figure(before: dict[str, dict[str, object]], after: dict[str, dict[str, object]]) -> None:
    focus_models = ["BODY_25B", "BODY_135"]
    fig, axes = plt.subplots(len(focus_models), 2, figsize=(13, 6.8))
    for row_idx, model in enumerate(focus_models):
        for col_idx, stage in enumerate(["before", "after"]):
            payload = before[model] if stage == "before" else after[model]
            axis = axes[row_idx, col_idx]
            axis.axis("off")
            axis.text(0.01, 0.92, f"{model} {stage}", fontsize=12, fontweight="bold", va="top", ha="left")
            axis.text(0.01, 0.76, wrap_markers(payload["markers"]), fontsize=9.5, va="top", ha="left", family="monospace")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_header_diff.png", dpi=200)
    plt.close(fig)


def build_markdown(before: dict[str, dict[str, object]], after: dict[str, dict[str, object]], presence_df: pd.DataFrame, count_df: pd.DataFrame, diff_text: str) -> str:
    summary_rows = []
    for model in MODELS:
        summary_rows.append(
            {
                "pose_model": model,
                "before_markers": before[model]["marker_count"],
                "after_markers": after[model]["marker_count"],
                "before_REar": "yes" if before[model]["has_REar"] else "no",
                "after_REar": "yes" if after[model]["has_REar"] else "no",
                "before_HeadTop": "yes" if before[model]["has_HeadTop"] else "no",
                "after_HeadTop": "yes" if after[model]["has_HeadTop"] else "no",
            }
        )
    summary_table = pd.DataFrame(summary_rows).to_markdown(index=False)
    presence_table = presence_df.to_markdown()
    count_table = count_df.to_markdown(index=False)

    return "\n".join(
        [
            "# TRC 耳朵与头顶点修复报告",
            "",
            "## 结论",
            "",
            "- 我已经实际跑了修复前和修复后的 `Pose2Sim.triangulation.triangulate_all()`，输入保持一致，确认问题存在且修复有效。",
            "- 根因不是三角化数值算法把点算丢了，而是 `Pose2Sim/skeletons.py` 里 `COCO`、`BODY_25`、`BODY_25B`、`BODY_135` 的 skeleton 定义漏掉了官方已有的 `REye/LEye/REar/LEar`，而 `BODY_25B` 与 `BODY_135` 还没有把官方的 `HeadTop` 暴露到 TRC。",
            "- 修复后，同一份 2D JSON 重新三角化：",
            "  - `COCO` / `BODY_25`：TRC 新增双眼和双耳。",
            "  - `BODY_25B` / `BODY_135`：TRC 新增双眼、双耳，并新增 `HeadTop`；同时保留原来的 `Head` 兼容旧流程。",
            "",
            "## 复现实验",
            "",
            "- 输入：四机位、两帧、合成标定、合成 2D JSON。JSON 里显式包含头部相关点。",
            "- 运行入口：仓库现有 `Pose2Sim.triangulation.triangulate_all()`。",
            "- 比较对象：完全相同的输入，在修复前后各跑一次。",
            "",
            "### 图 1：修复前后关键点存在性",
            "",
            "![Presence](../figures/trc_head_face_fix/fig01_presence_before_after.png)",
            "",
            "### 图 2：TRC marker 数量变化",
            "",
            "![Counts](../figures/trc_head_face_fix/fig02_marker_count.png)",
            "",
            "### 图 3：BODY_25B / BODY_135 表头前后对比",
            "",
            "![Headers](../figures/trc_head_face_fix/fig03_header_diff.png)",
            "",
            "## 数值结果",
            "",
            summary_table,
            "",
            presence_table,
            "",
            count_table,
            "",
            "修复前后的原始 TRC 已保存：",
            "",
            "- `data/trc_head_face_fix/before/*.trc`",
            "- `data/trc_head_face_fix/after/*.trc`",
            "",
            "## Debug 过程",
            "",
            "- `triangulation.py` 用 `RenderTree(model)` 生成 `keypoints_ids` / `keypoints_names`，TRC 表头完全由这两个列表决定。",
            "- 因此只要 skeleton 没有定义某个节点，即使 2D JSON 里有该 triplet，三角化也不会读取，更不会写进 TRC。",
            "- 修复前：",
            f"  - `BODY_25` 没有 `REye/LEye/REar/LEar`。",
            f"  - `BODY_25B` 和 `BODY_135` 没有 `REye/LEye/REar/LEar/HeadTop`，并把 id 18 只暴露成 `Head`。",
            "- 修复方式：只改 `Pose2Sim/skeletons.py`，把官方已有 body-part 映射补进 tree；没有改三角化数值过程。",
            "",
            "代码 diff 摘录：",
            "",
            "```diff",
            diff_text,
            "```",
            "",
            "## 官方映射依据",
            "",
            f"- OpenPose BODY_25 官方输出顺序文档：{OPENPOSE_OUTPUT_DOC_URL}",
            f"- OpenPose `BODY_25B` / `BODY_135` 官方 body-part 映射源码：{OPENPOSE_BODY25B_BODY135_URL}",
            "",
            "从官方映射可以直接看到：",
            "",
            "- `BODY_25`：15/16/17/18 分别是 `REye/LEye/REar/LEar`。",
            "- `BODY_25B`：1/2/3/4/18 分别是 `LEye/REye/LEar/REar/HeadTop`。",
            "- `BODY_135`：同样把 1/2/3/4/18 定义为 `LEye/REye/LEar/REar/HeadTop`。",
            "",
            "## 影响范围",
            "",
            "- 这次修复直接影响 TRC 列名与导出的 3D 点集合。",
            "- `BODY_25B` 和 `BODY_135` 为了兼容现有 OpenSim 配置，继续保留了 `Head`，同时新增同坐标的 `HeadTop`。",
            "- 我没有改 OpenSim 的 marker XML / IK setup，所以这次提交聚焦在“TRC 缺点”问题本身。",
        ]
    )


def main() -> None:
    ensure_dirs()
    before = load_summary("before")
    after = load_summary("after")
    presence_df = build_presence_dataframe(before, after)
    count_df = build_count_dataframe(before, after)
    make_presence_figure(presence_df)
    make_count_figure(count_df)
    make_header_figure(before, after)
    diff_text = git_diff_excerpt()
    DOC_PATH.write_text(build_markdown(before, after, presence_df, count_df, diff_text) + "\n", encoding="utf-8")
    print(f"Wrote report to {DOC_PATH}")


if __name__ == "__main__":
    main()
