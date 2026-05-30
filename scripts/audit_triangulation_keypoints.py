#!/usr/bin/env python

from __future__ import annotations

import json
import shutil
import subprocess
import tempfile
import textwrap
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import toml
from anytree import RenderTree

from Pose2Sim.skeletons import COCO_133, COCO_133_WRIST, HALPE_26
from Pose2Sim.triangulation import triangulate_all


ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data" / "triangulation_keypoint_audit"
FIG_DIR = ROOT / "figures" / "triangulation_keypoint_audit"
DOC_PATH = ROOT / "docs" / "triangulation-keypoint-audit.md"

INTERESTING_MARKERS = ["REye", "LEye", "REar", "LEar", "Head", "HeadTop"]
EXTRA_HEADTOP_INDEX = 133
IMAGE_SIZE = (1920, 1080)

MODEL_SPECS = [
    ("Body_with_feet", "HALPE_26", HALPE_26),
    ("Whole_body_wrist", "COCO_133_WRIST", COCO_133_WRIST),
    ("Whole_body", "COCO_133", COCO_133),
]


def ensure_output_dirs() -> None:
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    FIG_DIR.mkdir(parents=True, exist_ok=True)


def keypoint_names(model) -> list[str]:
    return [node.name for _, _, node in RenderTree(model) if node.id is not None]


def keypoint_ids(model) -> list[int]:
    return [node.id for _, _, node in RenderTree(model) if node.id is not None]


def wrap_marker_list(markers: list[str], width: int = 52) -> str:
    joined = ", ".join(markers)
    return textwrap.fill(joined, width=width)


def build_camera_sections() -> dict[str, dict[str, object]]:
    fx = fy = 1100.0
    cx = IMAGE_SIZE[0] / 2.0
    cy = IMAGE_SIZE[1] / 2.0
    matrix = [[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]]
    distortions = [0.0, 0.0, 0.0, 0.0]
    camera_translations = [
        [0.00, 0.00, 0.00],
        [0.25, 0.00, 0.00],
        [0.00, 0.18, 0.00],
        [-0.20, 0.10, 0.00],
    ]

    cameras = {}
    for index, translation in enumerate(camera_translations, start=1):
        cameras[f"cam{index:02d}"] = {
            "name": f"cam{index:02d}",
            "size": list(IMAGE_SIZE),
            "matrix": matrix,
            "distortions": distortions,
            "rotation": [0.0, 0.0, 0.0],
            "translation": translation,
        }
    return cameras


def synthetic_world_points(frame_index: int) -> dict[int, np.ndarray]:
    points = {}
    for point_index in range(EXTRA_HEADTOP_INDEX + 1):
        x = -0.55 + 0.08 * (point_index % 10) + 0.01 * frame_index
        y = 0.25 + 0.045 * (point_index // 10) + 0.005 * frame_index
        z = 4.2 + 0.04 * (point_index % 5)
        points[point_index] = np.array([x, y, z], dtype=np.float64)
    return points


def project_points_to_triplets(
    world_points: dict[int, np.ndarray],
    camera_translation: list[float],
) -> list[float]:
    fx = fy = 1100.0
    cx = IMAGE_SIZE[0] / 2.0
    cy = IMAGE_SIZE[1] / 2.0
    tx, ty, tz = camera_translation
    triplets = []

    for point_index in range(EXTRA_HEADTOP_INDEX + 1):
        world_point = world_points[point_index]
        x_cam = world_point[0] + tx
        y_cam = world_point[1] + ty
        z_cam = world_point[2] + tz
        x_px = fx * x_cam / z_cam + cx
        y_px = fy * y_cam / z_cam + cy
        triplets.extend([float(x_px), float(y_px), 0.95])

    return triplets


def write_synthetic_trial(trial_dir: Path) -> None:
    pose_dir = trial_dir / "pose"
    pose_dir.mkdir(parents=True, exist_ok=True)
    camera_sections = build_camera_sections()

    for camera_name, camera_section in camera_sections.items():
        camera_pose_dir = pose_dir / f"{camera_name}_json"
        camera_pose_dir.mkdir(parents=True, exist_ok=True)

        for frame_index in range(2):
            world_points = synthetic_world_points(frame_index)
            triplets = project_points_to_triplets(world_points, camera_section["translation"])
            json_payload = {
                "version": 1.3,
                "people": [
                    {
                        "person_id": [-1],
                        "pose_keypoints_2d": triplets,
                        "face_keypoints_2d": [],
                        "hand_left_keypoints_2d": [],
                        "hand_right_keypoints_2d": [],
                        "pose_keypoints_3d": [],
                        "face_keypoints_3d": [],
                        "hand_left_keypoints_3d": [],
                        "hand_right_keypoints_3d": [],
                    }
                ],
            }
            json_path = camera_pose_dir / f"{camera_name}_{frame_index:06d}.json"
            with open(json_path, "w", encoding="utf-8") as file_obj:
                json.dump(json_payload, file_obj)


def write_calibration(session_dir: Path) -> Path:
    calibration_dir = session_dir / "calibration"
    calibration_dir.mkdir(parents=True, exist_ok=True)
    calibration_path = calibration_dir / "synthetic_calibration.toml"
    with open(calibration_path, "w", encoding="utf-8") as file_obj:
        toml.dump(build_camera_sections(), file_obj)
    return calibration_path


def load_base_config() -> dict:
    return toml.load(ROOT / "Pose2Sim" / "Demo_SinglePerson" / "Config.toml")


def build_config(trial_dir: Path, pose_model: str) -> dict:
    config = load_base_config()
    config["project"]["project_dir"] = str(trial_dir.resolve())
    config["project"]["frame_range"] = [0, 2]
    config["project"]["frame_rate"] = 30
    config["project"]["multi_person"] = False

    config["pose"]["pose_model"] = pose_model

    triangulation = config["triangulation"]
    triangulation["parallel_triangulation"] = False
    triangulation["make_c3d"] = False
    triangulation["min_chunk_size"] = 1
    triangulation["sections_to_keep"] = "all"
    triangulation["interp_if_gap_smaller_than"] = 1
    triangulation["interpolation"] = "none"
    triangulation["show_interp_indices"] = False
    triangulation["fill_large_gaps_with"] = "zeros"
    triangulation["reproj_error_threshold_triangulation"] = 50
    triangulation["likelihood_threshold_triangulation"] = 0.1

    return config


def parse_trc_markers(trc_path: Path) -> list[str]:
    with open(trc_path, "r", encoding="utf-8") as file_obj:
        header_lines = [next(file_obj).rstrip("\n") for _ in range(4)]
    return [marker for marker in header_lines[3].split("\t")[2::3] if marker]


def git_output(args: list[str]) -> str:
    completed = subprocess.run(
        args,
        cwd=ROOT,
        check=True,
        text=True,
        capture_output=True,
    )
    return completed.stdout.strip()


def run_single_model_audit(pose_model: str, skeleton_name: str, model) -> dict[str, object]:
    with tempfile.TemporaryDirectory(prefix=f"pose2sim-audit-{pose_model}-") as temp_dir:
        temp_root = Path(temp_dir)
        session_dir = temp_root / "Session"
        trial_dir = session_dir / "Trial_1"
        session_dir.mkdir(parents=True, exist_ok=True)
        (session_dir / "Config.toml").write_text("# synthetic session for audit\n", encoding="utf-8")
        write_calibration(session_dir)
        write_synthetic_trial(trial_dir)

        config = build_config(trial_dir, pose_model)
        triangulate_all(config)

        trc_path = next((trial_dir / "pose-3d").glob("*.trc"))
        copied_trc_path = DATA_DIR / f"{pose_model}.trc"
        shutil.copy2(trc_path, copied_trc_path)

        sample_json_src = trial_dir / "pose" / "cam01_json" / "cam01_000000.json"
        sample_json_dst = DATA_DIR / "input_sample.json"
        shutil.copy2(sample_json_src, sample_json_dst)

        markers = parse_trc_markers(copied_trc_path)
        skeleton_markers = keypoint_names(model)

    result = {
        "pose_model": pose_model,
        "skeleton_name": skeleton_name,
        "skeleton_marker_count": len(skeleton_markers),
        "trc_marker_count": len(markers),
        "skeleton_markers": skeleton_markers,
        "trc_markers": markers,
        "has_reye": "REye" in markers,
        "has_leye": "LEye" in markers,
        "has_rear": "REar" in markers,
        "has_lear": "LEar" in markers,
        "has_head": "Head" in markers,
        "has_headtop": any("headtop" == marker.lower() or "head_top" == marker.lower() for marker in markers),
        "trc_path": str(copied_trc_path.relative_to(ROOT)),
    }
    return result


def find_real_trial_candidates() -> list[dict[str, object]]:
    candidates = []
    for pose_dir in ROOT.rglob("pose"):
        if not pose_dir.is_dir():
            continue
        trial_dir = pose_dir.parent
        config_path = trial_dir / "Config.toml"
        json_files = sorted(pose_dir.rglob("*.json"))
        trc_files = sorted((trial_dir / "pose-3d").glob("*.trc")) if (trial_dir / "pose-3d").exists() else []
        if config_path.exists() or json_files or trc_files:
            candidates.append(
                {
                    "trial_dir": str(trial_dir.relative_to(ROOT)),
                    "has_config": config_path.exists(),
                    "json_count": len(json_files),
                    "trc_count": len(trc_files),
                }
            )
    return candidates


def build_presence_table(results: list[dict[str, object]]) -> pd.DataFrame:
    input_markers = {"REye", "LEye", "REar", "LEar", "HeadTop"}
    rows = [
        {"row": "2D JSON input", **{marker: int(marker in input_markers) for marker in INTERESTING_MARKERS}},
    ]
    for result in results:
        rows.append(
            {
                "row": result["pose_model"],
                "REye": int(result["has_reye"]),
                "LEye": int(result["has_leye"]),
                "REar": int(result["has_rear"]),
                "LEar": int(result["has_lear"]),
                "Head": int(result["has_head"]),
                "HeadTop": int(result["has_headtop"]),
            }
        )
    return pd.DataFrame(rows).set_index("row")


def make_pipeline_figure() -> None:
    fig, ax = plt.subplots(figsize=(12, 3.2))
    ax.axis("off")

    boxes = [
        (0.03, "2D JSON\ntriplets can contain\nextra points"),
        (0.30, "pose_model selects\nskeleton nodes\nvia RenderTree(model)"),
        (0.57, "keypoints_ids +\nkeypoints_names"),
        (0.82, "TRC header only writes\nthose selected markers"),
    ]

    for x_pos, text in boxes:
        rect = plt.Rectangle((x_pos, 0.28), 0.18, 0.42, fill=False, linewidth=2.0)
        ax.add_patch(rect)
        ax.text(x_pos + 0.09, 0.49, text, ha="center", va="center", fontsize=11)

    for x_start in (0.21, 0.48, 0.75):
        ax.annotate("", xy=(x_start + 0.07, 0.49), xytext=(x_start, 0.49), arrowprops={"arrowstyle": "->", "lw": 2.0})

    ax.text(
        0.5,
        0.1,
        "Observed rule in this repo: Body_with_feet => no ears; Whole_body_wrist => eyes but no ears; Whole_body => ears; no built-in model exposes HeadTop.",
        ha="center",
        va="center",
        fontsize=10.5,
    )
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig01_pipeline.png", dpi=200)
    plt.close(fig)


def make_presence_figure(presence_df: pd.DataFrame) -> None:
    values = presence_df.values
    fig, ax = plt.subplots(figsize=(8.8, 3.5))
    ax.imshow(values, cmap="Greens", vmin=0, vmax=1, aspect="auto")

    ax.set_xticks(np.arange(len(presence_df.columns)), labels=presence_df.columns, rotation=0)
    ax.set_yticks(np.arange(len(presence_df.index)), labels=presence_df.index)

    for row_idx in range(values.shape[0]):
        for col_idx in range(values.shape[1]):
            label = "yes" if values[row_idx, col_idx] else "no"
            ax.text(col_idx, row_idx, label, ha="center", va="center", color="black", fontsize=10)

    ax.set_title("Marker Presence in Input JSON vs Generated TRC")
    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig02_presence_matrix.png", dpi=200)
    plt.close(fig)


def make_header_figure(results: list[dict[str, object]]) -> None:
    fig, axes = plt.subplots(len(results), 1, figsize=(12, 6.3))
    if len(results) == 1:
        axes = [axes]

    for axis, result in zip(axes, results):
        axis.axis("off")
        title = f"{result['pose_model']} -> {result['skeleton_name']} -> {result['trc_marker_count']} TRC markers"
        body = wrap_marker_list(result["trc_markers"][:36])
        axis.text(0.01, 0.78, title, fontsize=12, fontweight="bold", va="top", ha="left")
        axis.text(0.01, 0.55, body, fontsize=10, va="top", ha="left", family="monospace")

    fig.tight_layout()
    fig.savefig(FIG_DIR / "fig03_trc_headers.png", dpi=200)
    plt.close(fig)


def build_markdown_report(
    results: list[dict[str, object]],
    presence_df: pd.DataFrame,
    real_trials: list[dict[str, object]],
    blame_output: str,
    log_output: str,
) -> str:
    result_df = pd.DataFrame(
        [
            {
                "pose_model": result["pose_model"],
                "skeleton": result["skeleton_name"],
                "trc_markers": result["trc_marker_count"],
                "REye": "yes" if result["has_reye"] else "no",
                "LEye": "yes" if result["has_leye"] else "no",
                "REar": "yes" if result["has_rear"] else "no",
                "LEar": "yes" if result["has_lear"] else "no",
                "Head": "yes" if result["has_head"] else "no",
                "HeadTop": "yes" if result["has_headtop"] else "no",
            }
            for result in results
        ]
    )

    if real_trials:
        real_trial_lines = "\n".join(
            f"- `{candidate['trial_dir']}`: config={candidate['has_config']}, json={candidate['json_count']}, trc={candidate['trc_count']}"
            for candidate in real_trials
        )
    else:
        real_trial_lines = "- 扫描结果为空：当前仓库里没有可直接审计的 `pose/pose-3d` 真实试次。"

    presence_table = presence_df.to_markdown()
    result_table = result_df.to_markdown(index=False)

    first_result = next(result for result in results if result["pose_model"] == "Body_with_feet")
    wrist_result = next(result for result in results if result["pose_model"] == "Whole_body_wrist")
    whole_result = next(result for result in results if result["pose_model"] == "Whole_body")

    return "\n".join(
        [
            "# 2D 关键点与 TRC 丢点审计报告",
            "",
            "## 结论",
            "",
            "- 这不是最近并行三角化改动把耳朵或头顶“删掉”了。当前仓库里，`trc` 输出哪些 marker，取决于 `pose_model` 对应的 skeleton。`make_trc()` 只会把上游选中的 `keypoints_names` 原样写进 TRC 表头。",
            f"- `Body_with_feet -> HALPE_26`：实际代码跑出来的 TRC 只有 **{first_result['trc_marker_count']}** 个 marker，没有 `REye/LEye/REar/LEar/HeadTop`，但有 `Head`。",
            f"- `Whole_body_wrist -> COCO_133_WRIST`：实际代码跑出来的 TRC 有 **{wrist_result['trc_marker_count']}** 个 marker，保留 `REye/LEye`，但仍然**没有耳朵**，也没有 `HeadTop`。",
            f"- `Whole_body -> COCO_133`：实际代码跑出来的 TRC 有 **{whole_result['trc_marker_count']}** 个 marker，`REye/LEye/REar/LEar` 都在，但仍然**没有 `HeadTop`**。",
            "- 当前内置 skeleton 里没有任何 `HeadTop` 节点；如果 2D JSON 里额外带了头顶 triplet，三角化阶段也会因为 skeleton 没有这个 ID/名字而忽略它。",
            "",
            "## 图示",
            "",
            "![Pipeline](../figures/triangulation_keypoint_audit/fig01_pipeline.png)",
            "",
            "![Presence](../figures/triangulation_keypoint_audit/fig02_presence_matrix.png)",
            "",
            "![Headers](../figures/triangulation_keypoint_audit/fig03_trc_headers.png)",
            "",
            "## 实际代码测试",
            "",
            "审计脚本直接调用仓库现有的 `Pose2Sim.triangulation.triangulate_all()`，不是手工拼 TRC 字符串。输入 JSON 固定包含：",
            "",
            "- `REye/LEye/REar/LEar` 四个额外 face triplet",
            "- 一个额外的 `HeadTop` triplet（追加在 COCO_133 最大索引之后）",
            "- 四台相机、两帧、合成标定与合成 2D 观测",
            "",
            "运行结果汇总：",
            "",
            presence_table,
            "",
            result_table,
            "",
            "保存下来的运行证据：",
            "",
            "- `data/triangulation_keypoint_audit/input_sample.json`",
            "- `data/triangulation_keypoint_audit/Body_with_feet.trc`",
            "- `data/triangulation_keypoint_audit/Whole_body_wrist.trc`",
            "- `data/triangulation_keypoint_audit/Whole_body.trc`",
            "",
            "## 代码链路证据",
            "",
            "- `Pose2Sim/triangulation.py` 会先把 `pose_model` 映射到 skeleton，然后用 `RenderTree(model)` 生成 `keypoints_ids` 和 `keypoints_names`。",
            "- 同文件里的 `make_trc()` 只根据传入的 `keypoints_names` 写 TRC 表头，不会再做一次“删耳朵/删头顶”的过滤。",
            "- `Pose2Sim/poseEstimation.py` 里，`Whole_body_wrist` 和 `Whole_body` 都走 `COCO_133` RTMLib 检测模型；但 `triangulation.py` 里 `Whole_body_wrist` 会映射成 `COCO_133_WRIST` skeleton。这意味着 2D 可视化里能看到更多 face 点，不等于 TRC 会把这些点都写出来。",
            "",
            "关键 `git blame`：",
            "",
            "```text",
            blame_output,
            "```",
            "",
            "最近提交历史（截断）：",
            "",
            "```text",
            log_output,
            "```",
            "",
            "这说明核心 marker 映射行已经存在较久，不是这次并行三角化改动才引入的。",
            "",
            "## 真实 trial 扫描",
            "",
            real_trial_lines,
            "",
            "当前工作区没有用户真实 `pose/pose-3d` 试次可供逐文件核对，所以这份报告的运行证据全部来自可复现最小样例。它足够证明当前仓库代码的行为，但不能替代你本地那份真实 trial 的输入输出审计。",
            "",
            "## 建议",
            "",
            "- 如果你要在 TRC 里保留耳朵，不要用 `Body_with_feet`，也不要用 `Whole_body_wrist`；应改成 `pose_model = 'Whole_body'`，或者自定义 skeleton。",
            "- 如果你要在 TRC 里保留 `HeadTop`，当前内置模型都不行。需要用 `pose.CUSTOM` 明确定义这个节点，并同步更新 OpenSim marker/下游流程。",
            "- 如果你的 2D JSON 来自外部检测器，且里面已经有耳朵/头顶，但 Config 仍写成 `Body_with_feet` 或 `Whole_body_wrist`，那么这些额外 triplet 会在三角化阶段被忽略。这是配置与 skeleton 不匹配，不是最近三角化代码回归。",
        ]
    )


def main() -> None:
    ensure_output_dirs()

    results = [run_single_model_audit(*spec) for spec in MODEL_SPECS]
    presence_df = build_presence_table(results)
    presence_df.to_csv(DATA_DIR / "marker_presence.csv")

    real_trials = find_real_trial_candidates()
    with open(DATA_DIR / "real_trial_scan.json", "w", encoding="utf-8") as file_obj:
        json.dump(real_trials, file_obj, indent=2, ensure_ascii=False)

    summary = {
        "results": results,
        "real_trials": real_trials,
    }
    with open(DATA_DIR / "summary.json", "w", encoding="utf-8") as file_obj:
        json.dump(summary, file_obj, indent=2, ensure_ascii=False)

    make_pipeline_figure()
    make_presence_figure(presence_df)
    make_header_figure(results)

    blame_output = git_output(["git", "blame", "-L", "770,788", "Pose2Sim/triangulation.py"])
    log_output = git_output(["bash", "-lc", "git log --oneline -- Pose2Sim/triangulation.py | head -n 8"])

    report = build_markdown_report(results, presence_df, real_trials, blame_output, log_output)
    DOC_PATH.write_text(report + "\n", encoding="utf-8")

    print(f"Wrote report to {DOC_PATH}")


if __name__ == "__main__":
    main()
