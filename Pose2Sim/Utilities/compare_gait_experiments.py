#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import csv
import json
import math
import re
from pathlib import Path

import numpy as np
import pandas as pd


DETECTION_LOG_RE = re.compile(
    r"Person filter summary for (?P<camera>[^:]+): mean detections/frame "
    r"(?P<before>[0-9.]+) -> (?P<after>[0-9.]+), "
    r"frames with >1 detection (?P<multi_before>\d+) -> (?P<multi_after>\d+), "
    r"empty frames after filter (?P<empty_after>\d+)\."
)
DIRECT_PATH_RE = re.compile(r"Used the direct single-person association path on (?P<frames>\d+) frames\.")
ASSOC_NECK_RE = re.compile(
    r"Mean reprojection error for Neck point on all frames is (?P<error>[0-9.]+) px"
)
ASSOC_EXCLUDED_RE = re.compile(
    r"In average, (?P<excluded>[0-9.]+) cameras had to be excluded to reach the demanded 20 px error threshold"
)
TIME_RE = {
    "pose": re.compile(r"Pose estimation took (?P<time>\S+)\."),
    "sync": re.compile(r"Synchronization took (?P<time>\S+)\."),
    "assoc": re.compile(r"Associating persons took (?P<time>\S+)\."),
    "tri": re.compile(r"Triangulation took (?P<time>\S+)\."),
}
TRI_SUMMARY_RE = re.compile(
    r"Mean reprojection error for all points on frames .* is (?P<px>[0-9.]+) px, "
    r"which roughly corresponds to (?P<mm>[0-9.]+) mm"
)
TRI_EXCLUDED_RE = re.compile(r"In average, (?P<excluded>[0-9.]+) cameras had to be excluded")


def load_json_counts(json_dir):
    counts = []
    for json_path in sorted(json_dir.glob("*.json")):
        with open(json_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        counts.append(len(data.get("people", [])))
    return counts


def collect_detection_metrics(trial_dir):
    pose_dir = trial_dir / "pose"
    by_camera = {}
    combo_counts = []
    camera_dirs = sorted(pose_dir.glob("*_json"))
    camera_frame_counts = []
    for camera_dir in camera_dirs:
        counts = load_json_counts(camera_dir)
        camera_frame_counts.append(counts)
        by_camera[camera_dir.name.replace("_json", "")] = {
            "mean_detections": float(np.mean(counts)),
            "frames_multi": int(sum(c > 1 for c in counts)),
            "frames_zero": int(sum(c == 0 for c in counts)),
            "max_detections": int(max(counts) if counts else 0),
        }

    if camera_frame_counts:
        for per_frame in zip(*camera_frame_counts):
            combo = 1
            for count in per_frame:
                combo *= max(1, count)
            combo_counts.append(combo)

    summary = {
        "camera_metrics": by_camera,
        "mean_combo_candidates": float(np.mean(combo_counts)) if combo_counts else math.nan,
        "p95_combo_candidates": float(np.percentile(combo_counts, 95)) if combo_counts else math.nan,
        "max_combo_candidates": int(max(combo_counts) if combo_counts else 0),
    }
    return summary


def parse_log_metrics(log_paths):
    text = ""
    for log_path in log_paths:
        if log_path.exists():
            text += "\n" + log_path.read_text(encoding="utf-8")
    metrics = {
        "filter_summaries": {},
        "direct_single_person_frames": None,
        "assoc_neck_error_px": None,
        "assoc_avg_excluded_cams": None,
        "tri_mean_px": None,
        "tri_mean_mm": None,
        "tri_avg_excluded_cams": None,
        "times": {},
    }

    for match in DETECTION_LOG_RE.finditer(text):
        metrics["filter_summaries"][match.group("camera")] = {
            "before": float(match.group("before")),
            "after": float(match.group("after")),
            "multi_before": int(match.group("multi_before")),
            "multi_after": int(match.group("multi_after")),
            "empty_after": int(match.group("empty_after")),
        }

    direct_match = DIRECT_PATH_RE.search(text)
    if direct_match:
        metrics["direct_single_person_frames"] = int(direct_match.group("frames"))

    assoc_match = ASSOC_NECK_RE.search(text)
    if assoc_match:
        metrics["assoc_neck_error_px"] = float(assoc_match.group("error"))

    assoc_excluded_match = ASSOC_EXCLUDED_RE.search(text)
    if assoc_excluded_match:
        metrics["assoc_avg_excluded_cams"] = float(assoc_excluded_match.group("excluded"))

    tri_match = TRI_SUMMARY_RE.search(text)
    if tri_match:
        metrics["tri_mean_px"] = float(tri_match.group("px"))
        metrics["tri_mean_mm"] = float(tri_match.group("mm"))

    tri_excluded_matches = list(TRI_EXCLUDED_RE.finditer(text))
    if tri_excluded_matches:
        metrics["tri_avg_excluded_cams"] = float(tri_excluded_matches[-1].group("excluded"))

    for key, pattern in TIME_RE.items():
        match = pattern.search(text)
        if match:
            metrics["times"][key] = match.group("time")

    return metrics


def read_trc(path):
    lines = path.read_text(encoding="utf-8").splitlines()
    marker_line = lines[3].split("\t")
    markers = marker_line[2::3]
    markers = [marker for marker in markers if marker]
    cols = ["Frame#", "Time"]
    for marker in markers:
        cols.extend([f"{marker}_X", f"{marker}_Y", f"{marker}_Z"])
    df = pd.read_csv(path, sep="\t", skiprows=5, header=None)
    df = df.iloc[:, : len(cols)]
    df.columns = cols
    return df


def collect_jump_metrics(trc_path, markers):
    df = read_trc(trc_path)
    result = {}
    for marker in markers:
        arr = df[[f"{marker}_X", f"{marker}_Y", f"{marker}_Z"]].to_numpy(dtype=float)
        diffs = np.linalg.norm(np.diff(arr, axis=0), axis=1)
        result[marker] = {
            "mean_jump": float(np.nanmean(diffs)),
            "p95_jump": float(np.nanpercentile(diffs, 95)),
            "max_jump": float(np.nanmax(diffs)),
        }
    return result


def write_detection_association_report(output_path, experiments):
    lines = [
        "# 检测与 Association 对比",
        "",
        "## 2D 检测统计",
        "",
        "| 试次 | 相机 | 平均检测人数/帧 | 多人帧数 | 空帧数 | 最大检测人数 |",
        "| --- | --- | ---: | ---: | ---: | ---: |",
    ]
    for name, data in experiments.items():
        for camera, metrics in data["detection"]["camera_metrics"].items():
            lines.append(
                f"| {name} | {camera} | {metrics['mean_detections']:.2f} | "
                f"{metrics['frames_multi']} | {metrics['frames_zero']} | {metrics['max_detections']} |"
            )

    lines.extend([
        "",
        "## Association 统计",
        "",
        "| 试次 | 估计组合数均值 | 估计组合数 P95 | 估计组合数最大值 | Association 耗时 | 单人快路径帧数 | Neck 平均重投影误差(px) | Association 平均剔除相机数 |",
        "| --- | ---: | ---: | ---: | --- | ---: | ---: | ---: |",
    ])
    for name, data in experiments.items():
        assoc = data["log"]
        det = data["detection"]
        lines.append(
            f"| {name} | {det['mean_combo_candidates']:.2f} | {det['p95_combo_candidates']:.0f} | "
            f"{det['max_combo_candidates']} | {assoc['times'].get('assoc', 'NA')} | "
            f"{assoc['direct_single_person_frames'] if assoc['direct_single_person_frames'] is not None else 'NA'} | "
            f"{assoc['assoc_neck_error_px'] if assoc['assoc_neck_error_px'] is not None else 'NA'} | "
            f"{assoc['assoc_avg_excluded_cams'] if assoc['assoc_avg_excluded_cams'] is not None else 'NA'} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_triangulation_report(output_path, experiments, markers):
    lines = [
        "# 三角化与稳定性对比",
        "",
        "## 总体三角化指标",
        "",
        "| 试次 | 三角化耗时 | 平均重投影误差(px) | 平均重投影误差(mm) | 平均剔除相机数 |",
        "| --- | --- | ---: | ---: | ---: |",
    ]
    for name, data in experiments.items():
        log = data["log"]
        lines.append(
            f"| {name} | {log['times'].get('tri', 'NA')} | {log['tri_mean_px']} | {log['tri_mean_mm']} | {log['tri_avg_excluded_cams']} |"
        )

    baseline = experiments["gait_defaulttri"]["jump"]
    improved = experiments["gait_improvedtri"]["jump"]
    lines.extend([
        "",
        "## 默认三角化 vs 增强三角化 跳变对比",
        "",
        "| Marker | 默认 P95 跳变(m) | 增强 P95 跳变(m) | 变化量(m) | 默认最大跳变(m) | 增强最大跳变(m) |",
        "| --- | ---: | ---: | ---: | ---: | ---: |",
    ])
    for marker in markers:
        default_p95 = baseline[marker]["p95_jump"]
        improved_p95 = improved[marker]["p95_jump"]
        lines.append(
            f"| {marker} | {default_p95:.4f} | {improved_p95:.4f} | {improved_p95 - default_p95:+.4f} | "
            f"{baseline[marker]['max_jump']:.4f} | {improved[marker]['max_jump']:.4f} |"
        )

    output_path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_metrics_csv(output_path, experiments, markers):
    rows = []
    for name, data in experiments.items():
        row = {
            "trial": name,
            "mean_combo_candidates": data["detection"]["mean_combo_candidates"],
            "p95_combo_candidates": data["detection"]["p95_combo_candidates"],
            "max_combo_candidates": data["detection"]["max_combo_candidates"],
            "assoc_time": data["log"]["times"].get("assoc"),
            "direct_single_person_frames": data["log"]["direct_single_person_frames"],
            "assoc_neck_error_px": data["log"]["assoc_neck_error_px"],
            "assoc_avg_excluded_cams": data["log"]["assoc_avg_excluded_cams"],
            "tri_time": data["log"]["times"].get("tri"),
            "tri_mean_px": data["log"]["tri_mean_px"],
            "tri_mean_mm": data["log"]["tri_mean_mm"],
            "tri_avg_excluded_cams": data["log"]["tri_avg_excluded_cams"],
        }
        for camera, metrics in data["detection"]["camera_metrics"].items():
            prefix = camera.replace("-", "_")
            row[f"{prefix}_mean_detections"] = metrics["mean_detections"]
            row[f"{prefix}_frames_multi"] = metrics["frames_multi"]
            row[f"{prefix}_frames_zero"] = metrics["frames_zero"]
        for marker in markers:
            row[f"{marker}_p95_jump"] = data["jump"][marker]["p95_jump"]
            row[f"{marker}_max_jump"] = data["jump"][marker]["max_jump"]
        rows.append(row)

    fieldnames = sorted({key for row in rows for key in row.keys()})
    with open(output_path, "w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def infer_log_paths(root, trial_name):
    if trial_name == "gait_assoc_baseline":
        return [root / "logs" / "gait_assoc_baseline_postpose.log"]
    if trial_name == "gait_defaulttri":
        return [
            root / "logs" / "gait_defaulttri_postpose.log",
            root / "logs" / "gait_defaulttri_tri.log",
            root / "logs" / "gait_defaulttri_filter.log",
        ]
    if trial_name == "gait_improvedtri":
        return [root / "logs" / "gait_improvedtri_run.log"]
    raise ValueError(f"Unknown trial {trial_name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", required=True, help="Experiment root directory")
    args = parser.parse_args()

    root = Path(args.root).resolve()
    reports_dir = root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)

    markers = ["Head", "RWrist", "LWrist", "RAnkle", "LAnkle", "RBigToe", "LBigToe", "RHeel", "LHeel"]
    trial_names = ["gait_assoc_baseline", "gait_defaulttri", "gait_improvedtri"]
    experiments = {}

    for trial_name in trial_names:
        trial_dir = root / trial_name
        experiments[trial_name] = {
            "detection": collect_detection_metrics(trial_dir),
            "log": parse_log_metrics(infer_log_paths(root, trial_name)),
            "jump": collect_jump_metrics(
                trial_dir / "pose-3d" / f"{trial_name}_1-208_filt_butterworth.trc",
                markers,
            ),
        }

    write_detection_association_report(reports_dir / "comparison_detection_association.md", experiments)
    write_triangulation_report(reports_dir / "comparison_triangulation.md", experiments, markers)
    write_metrics_csv(reports_dir / "comparison_metrics.csv", experiments, markers)


if __name__ == "__main__":
    main()
