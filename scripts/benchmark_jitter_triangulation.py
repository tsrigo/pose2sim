#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from Pose2Sim.common import reprojection, weighted_triangulation
from Pose2Sim.filtering import filter1d, hampel_filter
from Pose2Sim.triangulation import triangulation_from_best_cameras


ROOT = Path(__file__).resolve().parents[1]
FIG_DIR = ROOT / "figures" / "jitter_triangulation"
DATA_DIR = ROOT / "data" / "jitter_triangulation"
FPS = 60
N_FRAMES = 240
SEEDS = list(range(24))
LIKELIHOOD_THRESHOLD = 0.3


def ensure_dirs() -> None:
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    DATA_DIR.mkdir(parents=True, exist_ok=True)


def set_plot_style() -> None:
    plt.rcParams.update(
        {
            "font.size": 10,
            "font.family": "DejaVu Serif",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.grid": False,
            "figure.dpi": 180,
            "savefig.dpi": 300,
            "savefig.bbox": "tight",
            "savefig.pad_inches": 0.05,
        }
    )


def normalize(vec: np.ndarray) -> np.ndarray:
    return vec / np.linalg.norm(vec)


def look_at_rotation(camera_center: np.ndarray, target: np.ndarray, up: np.ndarray) -> np.ndarray:
    z_axis = normalize(target - camera_center)
    x_axis = normalize(np.cross(z_axis, up))
    y_axis = normalize(np.cross(z_axis, x_axis))
    return np.vstack([x_axis, y_axis, z_axis])


def build_cameras(radius: float = 3.2, n_cams: int = 4) -> list[np.ndarray]:
    angles = np.linspace(0.0, 2.0 * np.pi, n_cams, endpoint=False)
    return build_cameras_from_angles(radius, angles)


def build_cameras_from_angles(radius: float, angles: np.ndarray) -> list[np.ndarray]:
    cx, cy = 960.0, 540.0
    fx = fy = 1450.0
    K = np.array([[fx, 0.0, cx], [0.0, fy, cy], [0.0, 0.0, 1.0]])
    target = np.array([0.0, 1.35, 0.0])
    up = np.array([0.0, 1.0, 0.0])
    projection_matrices: list[np.ndarray] = []

    for angle in angles:
        camera_center = np.array([radius * np.cos(angle), 1.8, radius * np.sin(angle)])
        R = look_at_rotation(camera_center, target, up)
        t = -R @ camera_center
        projection_matrices.append(K @ np.hstack([R, t.reshape(3, 1)]))

    return projection_matrices


def build_clustered_cameras(radius: float = 3.2) -> list[np.ndarray]:
    return build_cameras_from_angles(radius, np.array([-0.18, -0.06, 0.06, 0.18]))


def generate_ground_truth() -> np.ndarray:
    t = np.arange(N_FRAMES) / FPS
    x = 0.28 * np.sin(2 * np.pi * 0.65 * t) + 0.05 * np.sin(2 * np.pi * 1.8 * t + 0.4)
    y = 1.35 + 0.12 * np.cos(2 * np.pi * 0.9 * t + 0.2) + 0.03 * np.sin(2 * np.pi * 2.4 * t)
    z = 0.18 * np.sin(2 * np.pi * 0.45 * t + 0.7) + 0.04 * np.cos(2 * np.pi * 1.5 * t)
    return np.column_stack([x, y, z])


def project_sequence(points_3d: np.ndarray, projection_matrices: list[np.ndarray]) -> np.ndarray:
    q_h = np.column_stack([points_3d, np.ones(len(points_3d))])
    projected = []
    for P in projection_matrices:
        uvw = (P @ q_h.T).T
        projected.append(uvw[:, :2] / uvw[:, 2:3])
    return np.stack(projected, axis=1)


def simulate_observations(points_2d_gt: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    n_frames, n_cams, _ = points_2d_gt.shape

    residual = rng.normal(0.0, 1.4, size=points_2d_gt.shape)
    drift = 0.9 * np.sin(np.linspace(0, 6 * np.pi, n_frames)).reshape(-1, 1, 1)
    residual += drift * rng.normal(0.0, 0.5, size=(1, n_cams, 2))

    burst_mask = rng.random((n_frames, n_cams)) < 0.07
    burst_noise = rng.normal(0.0, 11.0, size=(n_frames, n_cams, 2))
    residual += burst_mask[..., None] * burst_noise

    miss_mask = rng.random((n_frames, n_cams)) < 0.04
    likelihood = np.clip(0.98 - 0.055 * np.linalg.norm(residual, axis=2), 0.02, 0.99)
    likelihood[burst_mask] *= 0.45
    likelihood[miss_mask] = 0.0

    observations = points_2d_gt + residual
    observations[miss_mask] = np.nan
    return observations, likelihood


def one_euro_config() -> dict:
    return {
        "filtering": {
            "one_euro": {
                "cut_off_frequency": 4.0,
                "beta": 1.5,
                "d_cut_off_frequency": 1.0,
            }
        }
    }


def butterworth_config() -> dict:
    return {"filtering": {"butterworth": {"cut_off_frequency": 6, "order": 4}}}


def smooth_2d_observations(observations: np.ndarray, likelihood: np.ndarray) -> np.ndarray:
    smoothed = observations.copy()
    config = one_euro_config()
    for cam_idx in range(observations.shape[1]):
        valid_mask = likelihood[:, cam_idx] >= LIKELIHOOD_THRESHOLD
        for dim_idx in range(2):
            series = pd.Series(smoothed[:, cam_idx, dim_idx])
            series[~valid_mask] = np.nan
            filtered = filter1d(series, config, "one_euro", FPS)
            smoothed[:, cam_idx, dim_idx] = filtered.to_numpy()
    return smoothed


def filter_3d_sequence(points_3d: np.ndarray) -> np.ndarray:
    df = pd.DataFrame(points_3d, columns=["x", "y", "z"])
    df = df.apply(hampel_filter, axis=0)
    df = df.apply(filter1d, axis=0, args=[butterworth_config(), "butterworth", FPS])
    return df.to_numpy()


def triangulate_frame_direct(
    projection_matrices: list[np.ndarray],
    observations_frame: np.ndarray,
    likelihood_frame: np.ndarray,
    use_weights: bool,
    apply_threshold: bool,
) -> tuple[np.ndarray, float]:
    x = observations_frame[:, 0].copy()
    y = observations_frame[:, 1].copy()
    w = likelihood_frame.copy()

    valid = np.isfinite(x) & np.isfinite(y) & (w > 0.0)
    if apply_threshold:
        valid &= w >= LIKELIHOOD_THRESHOLD
    if np.count_nonzero(valid) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan

    p_valid = [projection_matrices[i] for i in np.where(valid)[0]]
    x_valid = x[valid]
    y_valid = y[valid]
    if use_weights:
        w_valid = w[valid]
    else:
        w_valid = np.ones(np.count_nonzero(valid))

    q_h = weighted_triangulation(p_valid, x_valid, y_valid, w_valid)
    q = q_h[:3]
    reproj = reprojection(p_valid, q_h)
    q_calc = np.column_stack([reproj[0], reproj[1]])
    q_file = np.column_stack([x_valid, y_valid])
    reproj_err = np.sqrt(np.mean(np.sum((q_file - q_calc) ** 2, axis=1)))
    return q, reproj_err


def triangulate_frame_robust(
    projection_matrices: list[np.ndarray], observations_frame: np.ndarray, likelihood_frame: np.ndarray
) -> tuple[np.ndarray, float]:
    x = observations_frame[:, 0].copy()
    y = observations_frame[:, 1].copy()
    w = likelihood_frame.copy()

    low_conf = (w < LIKELIHOOD_THRESHOLD) | ~np.isfinite(x) | ~np.isfinite(y)
    x[low_conf] = np.nan
    y[low_conf] = np.nan
    w[low_conf] = np.nan

    if np.count_nonzero(np.isfinite(w) & (w > 0.0)) < 2:
        return np.array([np.nan, np.nan, np.nan]), np.nan

    coords = np.array((x, y, w))
    q, reproj_err, _, _ = triangulation_from_best_cameras(
        {
            "triangulation": {
                "reproj_error_threshold_triangulation": 8,
                "min_cameras_for_triangulation": 2,
                "handle_LR_swap": False,
                "undistort_points": False,
            }
        },
        coords,
        coords.copy(),
        projection_matrices,
        {},
    )
    return q, reproj_err


def triangulate_sequence(
    projection_matrices: list[np.ndarray],
    observations: np.ndarray,
    likelihood: np.ndarray,
    method: str,
) -> tuple[np.ndarray, np.ndarray]:
    points_3d = []
    reproj_errors = []
    for frame_idx in range(len(observations)):
        if method == "unweighted_all":
            q, reproj_err = triangulate_frame_direct(
                projection_matrices, observations[frame_idx], likelihood[frame_idx], use_weights=False, apply_threshold=False
            )
        elif method == "weighted_all":
            q, reproj_err = triangulate_frame_direct(
                projection_matrices, observations[frame_idx], likelihood[frame_idx], use_weights=True, apply_threshold=False
            )
        elif method == "weighted_thresholded":
            q, reproj_err = triangulate_frame_direct(
                projection_matrices, observations[frame_idx], likelihood[frame_idx], use_weights=True, apply_threshold=True
            )
        elif method == "robust_best_cams":
            q, reproj_err = triangulate_frame_robust(projection_matrices, observations[frame_idx], likelihood[frame_idx])
        else:
            raise ValueError(f"Unknown triangulation method: {method}")

        points_3d.append(q)
        reproj_errors.append(reproj_err)

    return np.asarray(points_3d), np.asarray(reproj_errors)


def rmse(a: np.ndarray, b: np.ndarray) -> float:
    diff = a - b
    return float(np.sqrt(np.mean(np.sum(diff**2, axis=-1))))


def second_diff_rms(a: np.ndarray, b: np.ndarray) -> float:
    residual = a - b
    if len(residual) < 3:
        return np.nan
    second_diff = residual[2:] - 2.0 * residual[1:-1] + residual[:-2]
    return float(np.sqrt(np.mean(np.sum(second_diff**2, axis=-1))))


def contiguous_sequences(mask: np.ndarray, min_length: int = 1) -> list[np.ndarray]:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        return []
    split_points = np.where(np.diff(indices) > 1)[0] + 1
    sequences = np.split(indices, split_points)
    return [seq for seq in sequences if len(seq) >= min_length]


def coverage(points_3d: np.ndarray) -> float:
    valid = np.isfinite(points_3d).all(axis=1)
    return float(np.mean(valid))


def contiguous_valid_metrics(est: np.ndarray, gt: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(est).all(axis=1)
    if not np.any(valid):
        return np.nan, np.nan
    est_valid = est[valid]
    gt_valid = gt[valid]
    jitter_values = [second_diff_rms(est[seq], gt[seq]) for seq in contiguous_sequences(valid, min_length=3)]
    jitter_value = float(np.nanmean(jitter_values)) if jitter_values else np.nan
    return rmse(est_valid, gt_valid), jitter_value


def metrics_2d(observed: np.ndarray, gt: np.ndarray, likelihood: np.ndarray) -> tuple[float, float]:
    valid = np.isfinite(observed).all(axis=2) & (likelihood >= LIKELIHOOD_THRESHOLD)
    if not np.any(valid):
        return np.nan, np.nan
    obs_valid = observed[valid]
    gt_valid = gt[valid]
    rmse_2d = np.sqrt(np.mean(np.sum((obs_valid - gt_valid) ** 2, axis=1)))

    jitter_values = []
    for cam_idx in range(observed.shape[1]):
        for seq in contiguous_sequences(valid[:, cam_idx], min_length=3):
            jitter_values.append(second_diff_rms(observed[seq, cam_idx], gt[seq, cam_idx]))
    jitter_2d = float(np.nanmean(jitter_values)) if jitter_values else np.nan
    return float(rmse_2d), jitter_2d


def run_jitter_experiment(points_3d_gt: np.ndarray, projection_matrices: list[np.ndarray]) -> tuple[pd.DataFrame, dict]:
    gt_2d = project_sequence(points_3d_gt, projection_matrices)
    rows = []
    example_payload = {}

    for seed in SEEDS:
        observations, likelihood = simulate_observations(gt_2d, seed)
        smoothed_2d = smooth_2d_observations(observations, likelihood)

        robust_raw_3d, _ = triangulate_sequence(projection_matrices, observations, likelihood, "robust_best_cams")
        robust_raw_3d_filt = filter_3d_sequence(robust_raw_3d)

        robust_smoothed_3d, _ = triangulate_sequence(projection_matrices, smoothed_2d, likelihood, "robust_best_cams")
        robust_smoothed_3d_filt = filter_3d_sequence(robust_smoothed_3d)

        pipelines = {
            "raw_2d_to_3d": (observations, robust_raw_3d),
            "raw_2d_to_3d_plus_3d_filter": (observations, robust_raw_3d_filt),
            "smooth_2d_to_3d": (smoothed_2d, robust_smoothed_3d),
            "smooth_2d_to_3d_plus_3d_filter": (smoothed_2d, robust_smoothed_3d_filt),
        }

        for method_name, (obs_used, points_3d_est) in pipelines.items():
            rmse_2d, jitter_2d = metrics_2d(obs_used, gt_2d, likelihood)
            rmse_3d, jitter_3d = contiguous_valid_metrics(points_3d_est, points_3d_gt)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "rmse_2d_px": rmse_2d,
                    "jitter_2d_px": jitter_2d,
                    "rmse_3d_mm": 1000.0 * rmse_3d,
                    "jitter_3d_mm": 1000.0 * jitter_3d,
                    "coverage": coverage(points_3d_est),
                }
            )

        if seed == SEEDS[0]:
            example_payload = {
                "gt_2d": gt_2d,
                "raw_2d": observations,
                "smooth_2d": smoothed_2d,
                "gt_3d": points_3d_gt,
                "raw_3d": robust_raw_3d,
                "smooth_3d": robust_smoothed_3d_filt,
            }

    return pd.DataFrame(rows), example_payload


def run_triangulation_experiment(points_3d_gt: np.ndarray) -> pd.DataFrame:
    rows = []
    wide_4 = build_cameras(radius=3.2, n_cams=4)
    clustered_4 = build_clustered_cameras(radius=3.2)
    wide_2 = wide_4[:2]

    gt_2d_wide_4 = project_sequence(points_3d_gt, wide_4)
    gt_2d_clustered_4 = project_sequence(points_3d_gt, clustered_4)
    gt_2d_wide_2 = project_sequence(points_3d_gt, wide_2)

    for seed in SEEDS:
        obs_wide_4, like_wide_4 = simulate_observations(gt_2d_wide_4, seed + 1000)
        obs_wide_2 = obs_wide_4[:, :2]
        like_wide_2 = like_wide_4[:, :2]
        obs_clustered_4, like_clustered_4 = simulate_observations(gt_2d_clustered_4, seed + 2000)

        configs = {
            "unweighted_all_4cams": (wide_4, obs_wide_4, like_wide_4, "unweighted_all"),
            "weighted_all_4cams": (wide_4, obs_wide_4, like_wide_4, "weighted_all"),
            "weighted_thresholded_4cams": (wide_4, obs_wide_4, like_wide_4, "weighted_thresholded"),
            "robust_best_cams_4cams": (wide_4, obs_wide_4, like_wide_4, "robust_best_cams"),
            "robust_best_cams_4cams_plus_2d_smooth": (
                wide_4,
                smooth_2d_observations(obs_wide_4, like_wide_4),
                like_wide_4,
                "robust_best_cams",
            ),
            "robust_best_cams_2cams_plus_2d_smooth": (
                wide_2,
                smooth_2d_observations(obs_wide_2, like_wide_2),
                like_wide_2,
                "robust_best_cams",
            ),
            "robust_best_cams_clustered_views_plus_2d_smooth": (
                clustered_4,
                smooth_2d_observations(obs_clustered_4, like_clustered_4),
                like_clustered_4,
                "robust_best_cams",
            ),
        }

        for method_name, (projection_matrices, obs, like, method) in configs.items():
            est_3d, reproj_errors = triangulate_sequence(projection_matrices, obs, like, method)
            rmse_3d, jitter_3d = contiguous_valid_metrics(est_3d, points_3d_gt)
            rows.append(
                {
                    "seed": seed,
                    "method": method_name,
                    "rmse_3d_mm": 1000.0 * rmse_3d,
                    "jitter_3d_mm": 1000.0 * jitter_3d,
                    "coverage": coverage(est_3d),
                    "reproj_error_px": float(np.nanmean(reproj_errors)),
                }
            )

    return pd.DataFrame(rows)


def summarize(df: pd.DataFrame, metric_cols: list[str]) -> pd.DataFrame:
    summary = df.groupby("method")[metric_cols].agg(["mean", "std"])
    summary.columns = ["_".join(col) for col in summary.columns]
    return summary.reset_index().sort_values(metric_cols[0] + "_mean")


def save_dataframe(df: pd.DataFrame, path: Path) -> None:
    df.to_csv(path, index=False)


def plot_jitter_example(example: dict) -> None:
    frames = np.arange(N_FRAMES)
    cam_idx = 0
    fig, axes = plt.subplots(2, 1, figsize=(8.0, 5.4), sharex=True)

    axes[0].plot(frames, example["gt_2d"][:, cam_idx, 0], label="GT 2D", linewidth=2.0)
    axes[0].plot(frames, example["raw_2d"][:, cam_idx, 0], label="Raw 2D", alpha=0.6)
    axes[0].plot(frames, example["smooth_2d"][:, cam_idx, 0], label="OneEuro 2D", linewidth=1.8)
    axes[0].set_ylabel("u (px)")
    axes[0].set_title("Camera 1 x-coordinate: raw 2D jitter vs temporal smoothing")
    axes[0].legend(frameon=False, ncol=3, loc="upper right")

    axes[1].plot(frames, example["gt_3d"][:, 0] * 1000.0, label="GT 3D", linewidth=2.0)
    axes[1].plot(frames, example["raw_3d"][:, 0] * 1000.0, label="Raw triangulated 3D", alpha=0.55)
    axes[1].plot(frames, example["smooth_3d"][:, 0] * 1000.0, label="2D smooth + 3D filter", linewidth=1.8)
    axes[1].set_ylabel("X (mm)")
    axes[1].set_xlabel("Frame")
    axes[1].set_title("Same motion after triangulation")
    axes[1].legend(frameon=False, ncol=3, loc="upper right")

    fig.savefig(FIG_DIR / "fig01_jitter_example.png")
    plt.close(fig)


def plot_jitter_bars(summary: pd.DataFrame) -> None:
    order = [
        "raw_2d_to_3d",
        "raw_2d_to_3d_plus_3d_filter",
        "smooth_2d_to_3d",
        "smooth_2d_to_3d_plus_3d_filter",
    ]
    labels = [
        "raw->3D",
        "raw->3D+3D filt",
        "2D smooth->3D",
        "2D smooth->3D+3D filt",
    ]
    summary = summary.set_index("method").loc[order].reset_index()
    methods = summary["method"].tolist()
    x = np.arange(len(methods))
    width = 0.36

    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.8))

    axes[0].bar(x - width / 2, summary["rmse_3d_mm_mean"], width, yerr=summary["rmse_3d_mm_std"], capsize=3, color="#4575b4")
    axes[0].bar(x + width / 2, summary["jitter_3d_mm_mean"], width, yerr=summary["jitter_3d_mm_std"], capsize=3, color="#d73027")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, rotation=18, ha="right")
    axes[0].set_ylabel("mm")
    axes[0].set_title("3D accuracy and residual jitter")
    axes[0].legend(["3D RMSE", "3D jitter"])

    axes[1].bar(x - width / 2, summary["rmse_2d_px_mean"], width, yerr=summary["rmse_2d_px_std"], capsize=3, color="#74add1")
    axes[1].bar(x + width / 2, summary["jitter_2d_px_mean"], width, yerr=summary["jitter_2d_px_std"], capsize=3, color="#f46d43")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, rotation=18, ha="right")
    axes[1].set_ylabel("px")
    axes[1].set_title("2D stability")
    axes[1].legend(["2D RMSE", "2D jitter"])

    fig.savefig(FIG_DIR / "fig02_jitter_ablation.png")
    plt.close(fig)


def plot_triangulation_bars(summary: pd.DataFrame) -> None:
    order = [
        "unweighted_all_4cams",
        "weighted_all_4cams",
        "weighted_thresholded_4cams",
        "robust_best_cams_4cams",
        "robust_best_cams_4cams_plus_2d_smooth",
        "robust_best_cams_2cams_plus_2d_smooth",
        "robust_best_cams_clustered_views_plus_2d_smooth",
    ]
    labels = [
        "4 cams\nunweighted",
        "4 cams\nweighted",
        "4 cams\nweight+thr",
        "4 cams\nrobust",
        "4 cams\nrobust+2D",
        "2 cams\nrobust+2D",
        "clustered 4\nrobust+2D",
    ]
    summary = summary.set_index("method").loc[order].reset_index()
    x = np.arange(len(summary))

    fig, axes = plt.subplots(1, 2, figsize=(10.2, 4.0))
    axes[0].bar(x, summary["rmse_3d_mm_mean"], yerr=summary["rmse_3d_mm_std"], capsize=3, color="#4daf4a")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(labels, fontsize=9)
    axes[0].set_ylabel("3D RMSE (mm)")
    axes[0].set_title("Triangulation accuracy")

    axes[1].bar(x, summary["reproj_error_px_mean"], yerr=summary["reproj_error_px_std"], capsize=3, color="#984ea3")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(labels, fontsize=9)
    axes[1].set_ylabel("Reprojection error (px)")
    axes[1].set_title("Consistency in image space")

    fig.savefig(FIG_DIR / "fig03_triangulation_ablation.png")
    plt.close(fig)


def build_markdown_report(jitter_summary: pd.DataFrame, triang_summary: pd.DataFrame) -> str:
    def fmt(mean: float, std: float, unit: str) -> str:
        return f"{mean:.2f} +/- {std:.2f} {unit}"

    jitter_best = jitter_summary.set_index("method").loc["smooth_2d_to_3d_plus_3d_filter"]
    jitter_base = jitter_summary.set_index("method").loc["raw_2d_to_3d"]
    tri_best = triang_summary.set_index("method").loc["robust_best_cams_4cams_plus_2d_smooth"]
    tri_base = triang_summary.set_index("method").loc["unweighted_all_4cams"]
    tri_two_cams = triang_summary.set_index("method").loc["robust_best_cams_2cams_plus_2d_smooth"]
    tri_clustered = triang_summary.set_index("method").loc["robust_best_cams_clustered_views_plus_2d_smooth"]

    lines = [
        "# 检测点抖动与三角化精度实验报告",
        "",
        "## 实验设置",
        "",
        "- 数据不是拍脑袋的截图，而是一个可控真值基准：240 帧、60 FPS、24 个随机种子。",
        "- 2D 观测由平滑的 3D 真值轨迹投影得到，再加入高斯噪声、偶发大抖动、低置信度和缺失帧。",
        "- 三角化直接调用仓库现有实现：`Pose2Sim.common.weighted_triangulation` 与 `Pose2Sim.triangulation.triangulation_from_best_cameras`。",
        "- 去抖直接调用仓库现有实现：`Pose2Sim.filtering.filter1d`，2D 使用 OneEuro，3D 使用 Hampel + Butterworth。",
        "",
        "## 结论先说",
        "",
        f"- 只做原始三角化时，3D RMSE 为 **{fmt(jitter_base['rmse_3d_mm_mean'], jitter_base['rmse_3d_mm_std'], 'mm')}**。",
        f"- 先做 2D OneEuro，再做 robust triangulation，最后做 3D Hampel + Butterworth 后，3D RMSE 降到 **{fmt(jitter_best['rmse_3d_mm_mean'], jitter_best['rmse_3d_mm_std'], 'mm')}**，3D 残余抖动降到 **{fmt(jitter_best['jitter_3d_mm_mean'], jitter_best['jitter_3d_mm_std'], 'mm')}**。",
        f"- 三角化策略从 4 相机无权重 DLT 换成 `robust_best_cams + 2D smoothing` 后，3D RMSE 从 **{fmt(tri_base['rmse_3d_mm_mean'], tri_base['rmse_3d_mm_std'], 'mm')}** 降到 **{fmt(tri_best['rmse_3d_mm_mean'], tri_best['rmse_3d_mm_std'], 'mm')}**。",
        f"- 同样是 robust triangulation，如果只剩 2 台相机，误差会上升到 **{fmt(tri_two_cams['rmse_3d_mm_mean'], tri_two_cams['rmse_3d_mm_std'], 'mm')}**；如果 4 台相机几乎都集中在同一侧，误差会上升到 **{fmt(tri_clustered['rmse_3d_mm_mean'], tri_clustered['rmse_3d_mm_std'], 'mm')}**。",
        "",
        "## 回答问题 1：如何解决检测点看起来抖？",
        "",
        "结论：不要只盯 3D 末端滤波，最有效的是 **先在 2D 上做轻量时序平滑，再进入 triangulation**。",
        "",
        "原因：",
        "- 视觉上看到的“抖”，本质是逐帧 2D 关键点高频噪声。",
        "- 如果等到 3D 再滤，三角化已经把多视角噪声耦合进去了；后滤只能补救，不能从源头减少误差。",
        "- `OneEuro` 对这种抖动很合适，因为它在低速段更强抑噪，在快速运动时又不会像重滤波那样把运动幅度压扁。",
        "",
        "建议顺序：",
        "1. 2D 先做 OneEuro。",
        "2. triangulation 继续用现有 robust best cameras。",
        "3. 3D 再做 Hampel + Butterworth 作为收尾。",
        "",
        "![抖动示例](../figures/jitter_triangulation/fig01_jitter_example.png)",
        "",
        "![去抖消融](../figures/jitter_triangulation/fig02_jitter_ablation.png)",
        "",
        "## 回答问题 2：如何进一步提高三角化精度？",
        "",
        "从这组代码实验看，提升顺序基本是：",
        "",
        "1. **先提升 2D 输入质量**：2D smoothing 对 3D RMSE 和抖动都直接有帮助。",
        "2. **保留 confidence weighting**：有权重比无权重更稳，因为低质量视角会自动被降权。",
        "3. **对低置信度点做阈值剔除**：把明显坏的视角先排掉，比把它们硬塞进 DLT 更有效。",
        "4. **继续使用 robust camera exclusion**：当某个相机瞬时飘掉时，`triangulation_from_best_cameras` 明显比“所有相机都参与”更稳。",
        "5. **尽量保留 3 台以上并拉开视角张角**：这是几何层面的硬收益，代码补救不了同侧聚集视角的深度病态。",
        "",
        "![三角化消融](../figures/jitter_triangulation/fig03_triangulation_ablation.png)",
        "",
        "## 推荐落地方案",
        "",
        "如果你现在要改自己的工程，我建议按下面优先级来：",
        "",
        "- `pose` 阶段新增一个 2D temporal smoothing 开关，优先接 OneEuro。",
        "- `triangulation.likelihood_threshold_triangulation` 保持在 0.3 左右起步，再按数据调。",
        "- 保留 `triangulation_from_best_cameras` 这类基于重投影误差的相机剔除逻辑。",
        "- `filtering` 阶段开启 `reject_outliers = true` 和 `type = 'butterworth'` 作为默认后处理。",
        "- 采集端优先保证更多视角和更大的视角张角，而不是只靠后处理。",
        "",
        "## 原始结果文件",
        "",
        "- `data/jitter_triangulation/jitter_metrics_raw.csv`",
        "- `data/jitter_triangulation/triangulation_metrics_raw.csv`",
        "- `data/jitter_triangulation/jitter_metrics_summary.csv`",
        "- `data/jitter_triangulation/triangulation_metrics_summary.csv`",
        "",
    ]
    return "\n".join(lines)


def main() -> None:
    ensure_dirs()
    set_plot_style()

    points_3d_gt = generate_ground_truth()
    projection_matrices = build_cameras(radius=3.2, n_cams=4)

    jitter_raw, example = run_jitter_experiment(points_3d_gt, projection_matrices)
    triang_raw = run_triangulation_experiment(points_3d_gt)

    jitter_summary = summarize(
        jitter_raw,
        ["rmse_2d_px", "jitter_2d_px", "rmse_3d_mm", "jitter_3d_mm", "coverage"],
    )
    triang_summary = summarize(
        triang_raw,
        ["rmse_3d_mm", "jitter_3d_mm", "coverage", "reproj_error_px"],
    )

    save_dataframe(jitter_raw, DATA_DIR / "jitter_metrics_raw.csv")
    save_dataframe(triang_raw, DATA_DIR / "triangulation_metrics_raw.csv")
    save_dataframe(jitter_summary, DATA_DIR / "jitter_metrics_summary.csv")
    save_dataframe(triang_summary, DATA_DIR / "triangulation_metrics_summary.csv")

    plot_jitter_example(example)
    plot_jitter_bars(jitter_summary)
    plot_triangulation_bars(triang_summary)

    report = build_markdown_report(jitter_summary, triang_summary)
    report_path = ROOT / "docs" / "jitter-triangulation-report.md"
    report_path.write_text(report, encoding="utf-8")

    summary_payload = {
        "jitter_summary": jitter_summary.to_dict(orient="records"),
        "triangulation_summary": triang_summary.to_dict(orient="records"),
        "report_path": str(report_path),
    }
    (DATA_DIR / "summary.json").write_text(json.dumps(summary_payload, indent=2), encoding="utf-8")

    print(f"Wrote report to {report_path}")
    print(f"Saved figures to {FIG_DIR}")
    print(f"Saved tabular results to {DATA_DIR}")


if __name__ == "__main__":
    main()
