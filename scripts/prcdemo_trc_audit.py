#!/usr/bin/env python3
"""Audit and export TRC files for the PRCDemo Unity viewer.

PRCDemo's bundled TrcDataLoader maps source TRC coordinates to Unity as:
    Unity = (source X, -source Y, source Z)

The demo0417v6 Pose2Sim output is Y-down/up with the anatomical left-right
width primarily in source Z and forward/back primarily in source -X. For
PRCDemo visualization, export:
    X_prcdemo = Z_pose2sim
    Y_prcdemo = Y_pose2sim
    Z_prcdemo = -X_pose2sim
"""

from __future__ import annotations

import argparse
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

import numpy as np


PRCDEMO_BONE_PAIRS = (
    ("Hip", "RHip"),
    ("RHip", "RKnee"),
    ("RKnee", "RAnkle"),
    ("RAnkle", "RHeel"),
    ("RAnkle", "RBigToe"),
    ("RAnkle", "RSmallToe"),
    ("Hip", "LHip"),
    ("LHip", "LKnee"),
    ("LKnee", "LAnkle"),
    ("LAnkle", "LHeel"),
    ("LAnkle", "LBigToe"),
    ("LAnkle", "LSmallToe"),
    ("Hip", "Neck"),
    ("Neck", "Head"),
    ("Head", "Nose"),
    ("Neck", "RShoulder"),
    ("RShoulder", "RElbow"),
    ("RElbow", "RWrist"),
    ("Neck", "LShoulder"),
    ("LShoulder", "LElbow"),
    ("LElbow", "LWrist"),
)

AXIS_DIAGNOSTIC_PAIRS = (
    ("LHip", "RHip"),
    ("LKnee", "RKnee"),
    ("LAnkle", "RAnkle"),
    ("LShoulder", "RShoulder"),
)


@dataclass
class TrcData:
    path: Path
    header: list[str]
    frame_labels: list[str]
    time_labels: list[str]
    markers: list[str]
    points: np.ndarray  # frames x markers x xyz, source TRC coordinates


@dataclass
class AssessmentRow:
    variant: str
    item: str
    score: int | None
    metric_label: str
    metric_value: float | None
    unit: str
    start_frame: int
    window: int
    detail: str


def load_trc(path: Path) -> TrcData:
    lines = path.read_text(encoding="utf-8").splitlines()
    if len(lines) < 6:
        raise ValueError(f"TRC file is too short: {path}")

    marker_cells = lines[3].split("\t")[2:]
    markers = [cell.strip() for cell in marker_cells if cell.strip()]
    if not markers:
        raise ValueError(f"No marker names found in TRC header: {path}")

    frame_labels: list[str] = []
    time_labels: list[str] = []
    frames: list[np.ndarray] = []
    expected_values = 2 + len(markers) * 3

    for line in lines[5:]:
        if not line.strip():
            continue
        cells = line.split("\t")
        if len(cells) < expected_values:
            continue
        frame_labels.append(cells[0])
        time_labels.append(cells[1])
        values = [float(cell) if cell else math.nan for cell in cells[2:expected_values]]
        frames.append(np.asarray(values, dtype=float).reshape(len(markers), 3))

    if not frames:
        raise ValueError(f"No frame rows parsed from TRC file: {path}")

    return TrcData(
        path=path,
        header=lines[:5],
        frame_labels=frame_labels,
        time_labels=time_labels,
        markers=markers,
        points=np.stack(frames, axis=0),
    )


def pose2sim_to_prcdemo_source(points: np.ndarray) -> np.ndarray:
    """Return a source TRC coordinate array tailored to PRCDemo's axis map."""
    return np.stack((points[..., 2], points[..., 1], -points[..., 0]), axis=-1)


def prcdemo_source_to_unity(points: np.ndarray) -> np.ndarray:
    return np.stack((points[..., 0], -points[..., 1], points[..., 2]), axis=-1)


def format_float(value: float) -> str:
    if math.isnan(value):
        return ""
    return format(float(value), ".15g")


def write_trc_like(source: TrcData, points: np.ndarray, output_path: Path) -> None:
    if points.shape != source.points.shape:
        raise ValueError("Transformed points must match source point shape.")

    header = list(source.header)
    path_parts = header[0].split("\t")
    if len(path_parts) >= 4:
        path_parts[3] = output_path.name
        header[0] = "\t".join(path_parts)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        for line in header:
            handle.write(line + "\n")
        for frame_label, time_label, frame in zip(
            source.frame_labels, source.time_labels, points, strict=True
        ):
            row = [frame_label, time_label]
            row.extend(format_float(value) for value in frame.reshape(-1))
            handle.write("\t".join(row) + "\n")


def clamp01(value: float) -> float:
    return max(0.0, min(1.0, value))


def score_lower_is_better(value: float, thresholds: tuple[float, float, float]) -> int:
    if value <= thresholds[0]:
        return 3
    if value <= thresholds[1]:
        return 2
    if value <= thresholds[2]:
        return 1
    return 0


def score_cva(value: float) -> int:
    if value >= 53.0:
        return 3
    if value >= 50.0:
        return 2
    if value >= 40.0:
        return 1
    return 0


def vector_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    a_norm = float(np.linalg.norm(a))
    b_norm = float(np.linalg.norm(b))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    cos_value = float(np.dot(a, b) / (a_norm * b_norm))
    return math.degrees(math.acos(max(-1.0, min(1.0, cos_value))))


def signed_angle_on_plane(a: np.ndarray, b: np.ndarray, normal: np.ndarray) -> float:
    normal = normal / max(float(np.linalg.norm(normal)), 1e-8)
    a_projected = a - normal * float(np.dot(a, normal))
    b_projected = b - normal * float(np.dot(b, normal))
    a_norm = float(np.linalg.norm(a_projected))
    b_norm = float(np.linalg.norm(b_projected))
    if a_norm < 1e-8 or b_norm < 1e-8:
        return 0.0
    a_unit = a_projected / a_norm
    b_unit = b_projected / b_norm
    signed_sin = float(np.dot(normal, np.cross(a_unit, b_unit)))
    signed_cos = float(np.dot(a_unit, b_unit))
    return math.degrees(math.atan2(signed_sin, signed_cos))


def find_most_stable_window_start(unity_points: np.ndarray, window: int) -> int:
    frame_count = unity_points.shape[0]
    window = max(1, min(int(window), frame_count))
    if frame_count <= window:
        return 0

    best_start = 0
    best_score = float("inf")
    for start in range(0, frame_count - window + 1):
        score = 0.0
        for frame_idx in range(start + 1, start + window):
            delta = unity_points[frame_idx] - unity_points[frame_idx - 1]
            score += float(np.linalg.norm(delta, axis=1).sum())
        if score < best_score:
            best_score = score
            best_start = start
    return best_start


class PrcDemoAssessment:
    def __init__(self, markers: list[str], source_points: np.ndarray, window: int):
        self.markers = markers
        self.marker_index = {name.lower(): idx for idx, name in enumerate(markers)}
        self.unity_points = prcdemo_source_to_unity(source_points)
        self.window = max(1, min(int(window), self.unity_points.shape[0]))
        self.start = find_most_stable_window_start(self.unity_points, self.window)

    def has(self, marker: str) -> bool:
        return marker.lower() in self.marker_index

    def average(self, marker: str) -> np.ndarray | None:
        idx = self.marker_index.get(marker.lower())
        if idx is None:
            return None
        end = min(self.start + self.window, self.unity_points.shape[0])
        return self.unity_points[self.start : end, idx, :].mean(axis=0)

    def require(self, *markers: str) -> list[np.ndarray] | None:
        values = [self.average(marker) for marker in markers]
        if any(value is None for value in values):
            return None
        return values  # type: ignore[return-value]

    def unavailable(self, variant: str, item: str, reason: str) -> AssessmentRow:
        return AssessmentRow(
            variant=variant,
            item=item,
            score=None,
            metric_label="unavailable",
            metric_value=None,
            unit="",
            start_frame=self.start,
            window=self.window,
            detail=reason,
        )

    def assess(self, variant: str) -> list[AssessmentRow]:
        return [
            self.shoulder_height_difference(variant),
            self.forward_head_posture(variant),
            self.head_tilt(variant),
            self.leg_length_load_bias(variant),
            self.bow_legs(variant),
            self.knock_knees(variant),
            self.knee_hyperextension(variant),
            self.foot_progression_static(variant),
        ]

    def shoulder_height_difference(self, variant: str) -> AssessmentRow:
        values = self.require("RShoulder", "LShoulder")
        if values is None:
            return self.unavailable(variant, "高低肩", "缺少肩部标记点")
        right, left = values
        diff_cm = abs(float(right[1] - left[1])) * 100.0
        return AssessmentRow(
            variant,
            "高低肩",
            score_lower_is_better(diff_cm, (0.5, 1.0, 2.0)),
            "肩峰高度差",
            diff_cm,
            "cm",
            self.start,
            self.window,
            f"LShoulder={left.tolist()}, RShoulder={right.tolist()}",
        )

    def forward_head_posture(self, variant: str) -> AssessmentRow:
        neck = self.average("Neck")
        face = self.average("Nose")
        if face is None:
            face = self.average("Head")
        if neck is None or face is None:
            return self.unavailable(variant, "颈前倾", "缺少头颈标记点")
        vector = face - neck
        cva = math.degrees(math.atan2(abs(float(vector[1])), max(abs(float(vector[2])), 1e-4)))
        return AssessmentRow(
            variant,
            "颈前倾",
            score_cva(cva),
            "CVA",
            cva,
            "deg",
            self.start,
            self.window,
            f"neck={neck.tolist()}, face={face.tolist()}, vector={vector.tolist()}",
        )

    def head_tilt(self, variant: str) -> AssessmentRow:
        neck = self.average("Neck")
        face = self.average("Nose")
        if face is None:
            face = self.average("Head")
        if neck is None or face is None:
            return self.unavailable(variant, "头侧倾", "缺少头部或颈部标记点")
        vector = face - neck
        angle = math.degrees(math.atan2(abs(float(vector[0])), max(abs(float(vector[1])), 1e-4)))
        return AssessmentRow(
            variant,
            "头侧倾",
            score_lower_is_better(angle, (2.0, 5.0, 8.0)),
            "倾斜角",
            angle,
            "deg",
            self.start,
            self.window,
            f"neck={neck.tolist()}, face={face.tolist()}, vector={vector.tolist()}",
        )

    def leg_length_load_bias(self, variant: str) -> AssessmentRow:
        values = self.require("LHip", "RHip", "LHeel", "RHeel", "LAnkle", "RAnkle")
        if values is None:
            return self.unavailable(variant, "长短腿/负重偏侧", "缺少下肢标记点")
        left_hip, right_hip, left_heel, right_heel, left_ankle, right_ankle = values
        left_foot_x = float(((left_heel + left_ankle) * 0.5)[0])
        right_foot_x = float(((right_heel + right_ankle) * 0.5)[0])
        pelvis_x = float(((left_hip + right_hip) * 0.5)[0])
        denominator = max(abs(right_foot_x - left_foot_x), 1e-4)
        right_load = clamp01((pelvis_x - left_foot_x) / denominator)
        left_load = 1.0 - right_load
        diff_percent = abs(left_load - right_load) * 100.0
        return AssessmentRow(
            variant,
            "长短腿/负重偏侧",
            score_lower_is_better(diff_percent, (5.0, 10.0, 15.0)),
            "左右负荷差",
            diff_percent,
            "%",
            self.start,
            self.window,
            (
                f"left_foot_x={left_foot_x:.6f}, right_foot_x={right_foot_x:.6f}, "
                f"pelvis_x={pelvis_x:.6f}, left_load={left_load * 100.0:.3f}%, "
                f"right_load={right_load * 100.0:.3f}%"
            ),
        )

    def bow_legs(self, variant: str) -> AssessmentRow:
        values = self.require("RKnee", "LKnee")
        if values is None:
            return self.unavailable(variant, "O型腿", "缺少膝关节标记点")
        right, left = values
        distance_cm = abs(float(right[0] - left[0])) * 100.0
        return AssessmentRow(
            variant,
            "O型腿",
            score_lower_is_better(distance_cm, (2.0, 4.0, 5.0)),
            "膝间距ICD",
            distance_cm,
            "cm",
            self.start,
            self.window,
            f"LKnee={left.tolist()}, RKnee={right.tolist()}",
        )

    def knock_knees(self, variant: str) -> AssessmentRow:
        values = self.require("RAnkle", "LAnkle")
        if values is None:
            return self.unavailable(variant, "X型腿", "缺少踝关节标记点")
        right, left = values
        distance_cm = abs(float(right[0] - left[0])) * 100.0
        return AssessmentRow(
            variant,
            "X型腿",
            score_lower_is_better(distance_cm, (2.5, 5.0, 7.0)),
            "踝间距IMD",
            distance_cm,
            "cm",
            self.start,
            self.window,
            f"LAnkle={left.tolist()}, RAnkle={right.tolist()}",
        )

    def knee_hyperextension(self, variant: str) -> AssessmentRow:
        left = self.knee_extension_angle("LHip", "LKnee", "LAnkle")
        right = self.knee_extension_angle("RHip", "RKnee", "RAnkle")
        max_angle = max(left, right)
        return AssessmentRow(
            variant,
            "膝超伸",
            score_lower_is_better(max_angle, (5.0, 10.0, 15.0)),
            "超伸角",
            max_angle,
            "deg",
            self.start,
            self.window,
            f"left={left:.6f} deg, right={right:.6f} deg",
        )

    def knee_extension_angle(self, hip_marker: str, knee_marker: str, ankle_marker: str) -> float:
        values = self.require(hip_marker, knee_marker, ankle_marker)
        if values is None:
            return 0.0
        hip, knee, ankle = values
        thigh = hip - knee
        shank = ankle - knee
        thigh[0] = 0.0
        shank[0] = 0.0
        return abs(180.0 - vector_angle_degrees(thigh, shank))

    def foot_progression_static(self, variant: str) -> AssessmentRow:
        left = self.foot_progression_angle("LHeel", "LBigToe", "LSmallToe")
        right = self.foot_progression_angle("RHeel", "RBigToe", "RSmallToe")
        max_abs_angle = max(abs(left), abs(right))
        if -3.0 <= left <= 20.0 and -3.0 <= right <= 20.0:
            score = 3
        elif -5.0 <= left <= 25.0 and -5.0 <= right <= 25.0:
            score = 2
        elif -8.0 <= left <= 30.0 and -8.0 <= right <= 30.0:
            score = 1
        else:
            score = 0
        return AssessmentRow(
            variant,
            "内外八字",
            score,
            "最大足进展角",
            max_abs_angle,
            "deg",
            self.start,
            self.window,
            f"left={left:.6f} deg, right={right:.6f} deg",
        )

    def foot_progression_angle(self, heel_marker: str, big_marker: str, small_marker: str) -> float:
        values = self.require(heel_marker, big_marker, small_marker)
        if values is None:
            return 0.0
        heel, big, small = values
        vector = (big + small) * 0.5 - heel
        return signed_angle_on_plane(np.array([0.0, 0.0, 1.0]), vector, np.array([0.0, 1.0, 0.0]))


def assessment_rows(trc: TrcData, window: int) -> list[AssessmentRow]:
    variants = {
        "current_trc_axes": trc.points,
        "prcdemo_visual_axes": pose2sim_to_prcdemo_source(trc.points),
    }
    rows: list[AssessmentRow] = []
    for variant, points in variants.items():
        rows.extend(PrcDemoAssessment(trc.markers, points, window).assess(variant))
    return rows


def write_assessment_csv(rows: Iterable[AssessmentRow], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "axis_variant",
                "item",
                "score",
                "metric_label",
                "metric_value",
                "unit",
                "start_frame",
                "window",
                "detail",
            ],
        )
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "axis_variant": row.variant,
                    "item": row.item,
                    "score": "" if row.score is None else row.score,
                    "metric_label": row.metric_label,
                    "metric_value": "" if row.metric_value is None else f"{row.metric_value:.6f}",
                    "unit": row.unit,
                    "start_frame": row.start_frame,
                    "window": row.window,
                    "detail": row.detail,
                }
            )


def axis_diagnostic_rows(trc: TrcData, window: int) -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    variants = {
        "current_trc_axes": trc.points,
        "prcdemo_visual_axes": pose2sim_to_prcdemo_source(trc.points),
    }
    for variant, source_points in variants.items():
        unity = prcdemo_source_to_unity(source_points)
        assessment = PrcDemoAssessment(trc.markers, source_points, window)
        start = assessment.start
        end = start + assessment.window
        for left_name, right_name in AXIS_DIAGNOSTIC_PAIRS:
            if not assessment.has(left_name) or not assessment.has(right_name):
                continue
            left = unity[start:end, assessment.marker_index[left_name.lower()], :].mean(axis=0)
            right = unity[start:end, assessment.marker_index[right_name.lower()], :].mean(axis=0)
            delta = right - left
            rows.append(
                {
                    "axis_variant": variant,
                    "pair": f"{left_name}-{right_name}",
                    "unity_x_span_cm": f"{abs(float(delta[0])) * 100.0:.6f}",
                    "unity_y_delta_cm": f"{abs(float(delta[1])) * 100.0:.6f}",
                    "unity_z_span_cm": f"{abs(float(delta[2])) * 100.0:.6f}",
                    "three_d_distance_cm": f"{float(np.linalg.norm(delta)) * 100.0:.6f}",
                    "start_frame": str(start),
                    "window": str(assessment.window),
                }
            )
    return rows


def write_axis_diagnostic_csv(rows: list[dict[str, str]], output_path: Path) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "axis_variant",
                "pair",
                "unity_x_span_cm",
                "unity_y_delta_cm",
                "unity_z_span_cm",
                "three_d_distance_cm",
                "start_frame",
                "window",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)


def write_axis_figure(rows: list[dict[str, str]], output_path: Path) -> None:
    import matplotlib.pyplot as plt

    pairs = list(dict.fromkeys(row["pair"] for row in rows))
    variants = ["current_trc_axes", "prcdemo_visual_axes"]
    width = 0.35
    x_positions = np.arange(len(pairs))

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for axis, span_key, title in [
        (axes[0], "unity_x_span_cm", "Unity X span used as left-right"),
        (axes[1], "unity_z_span_cm", "Unity Z span used as depth"),
    ]:
        for variant_idx, variant in enumerate(variants):
            values = []
            for pair in pairs:
                match = next(row for row in rows if row["axis_variant"] == variant and row["pair"] == pair)
                values.append(float(match[span_key]))
            axis.bar(x_positions + (variant_idx - 0.5) * width, values, width, label=variant)
        axis.set_title(title)
        axis.set_xticks(x_positions)
        axis.set_xticklabels(pairs, rotation=30, ha="right")
        axis.set_ylabel("cm")
        axis.grid(axis="y", alpha=0.25)
    axes[0].legend()
    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=160)
    plt.close(fig)


def summarize(rows: list[AssessmentRow], axis_rows: list[dict[str, str]]) -> str:
    lines: list[str] = []
    for variant in ["current_trc_axes", "prcdemo_visual_axes"]:
        variant_rows = [row for row in rows if row.variant == variant and row.score is not None]
        if variant_rows:
            average = sum(row.score or 0 for row in variant_rows) / len(variant_rows)
            lines.append(f"{variant}: average_score={average:.3f}/3 over {len(variant_rows)} static items")
            for row in variant_rows:
                lines.append(
                    f"  {row.item}: score={row.score}, {row.metric_label}="
                    f"{row.metric_value:.3f}{row.unit if row.unit != 'deg' else 'deg'}"
                )
    lines.append("Axis diagnostic:")
    for row in axis_rows:
        if row["pair"] in {"LHip-RHip", "LShoulder-RShoulder"}:
            lines.append(
                f"  {row['axis_variant']} {row['pair']}: "
                f"UnityX={float(row['unity_x_span_cm']):.2f}cm, "
                f"UnityZ={float(row['unity_z_span_cm']):.2f}cm, "
                f"3D={float(row['three_d_distance_cm']):.2f}cm"
            )
    return "\n".join(lines)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("trc", type=Path, help="Input Pose2Sim TRC file.")
    parser.add_argument("--window", type=int, default=30, help="Static stable window length in frames.")
    parser.add_argument("--out-trc", type=Path, help="Write a PRCDemo visual-axis TRC.")
    parser.add_argument("--metrics-csv", type=Path, help="Write PRCDemo-like static assessment CSV.")
    parser.add_argument("--axis-csv", type=Path, help="Write axis/proportion diagnostic CSV.")
    parser.add_argument("--figure", type=Path, help="Write axis/proportion diagnostic figure.")
    args = parser.parse_args()

    trc = load_trc(args.trc)
    if args.out_trc:
        write_trc_like(trc, pose2sim_to_prcdemo_source(trc.points), args.out_trc)

    rows = assessment_rows(trc, args.window)
    axis_rows = axis_diagnostic_rows(trc, args.window)
    if args.metrics_csv:
        write_assessment_csv(rows, args.metrics_csv)
    if args.axis_csv:
        write_axis_diagnostic_csv(axis_rows, args.axis_csv)
    if args.figure:
        write_axis_figure(axis_rows, args.figure)

    print(summarize(rows, axis_rows))


if __name__ == "__main__":
    main()
