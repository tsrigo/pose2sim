#!/usr/bin/env python
# -*- coding: utf-8 -*-

import argparse
import re
from io import StringIO
from pathlib import Path

import pandas as pd


def load_ik_errors(sto_path):
    lines = sto_path.read_text(encoding='utf-8').splitlines()
    start_idx = next(i for i, line in enumerate(lines) if line.strip() == 'endheader') + 1
    return pd.read_csv(StringIO('\n'.join(lines[start_idx:])), sep=r'\s+')


def parse_latest_triangulation_metrics(log_path):
    text = log_path.read_text(encoding='utf-8', errors='ignore')
    pattern = re.compile(
        r"Mean reprojection error for all points on frames (\d+) to (\d+) is ([0-9.]+) px, which roughly corresponds to ([0-9.]+) mm\.",
        re.MULTILINE,
    )
    matches = pattern.findall(text)
    if not matches:
        return None
    start_frame, end_frame, error_px, error_mm = matches[-1]
    return {
        'frame_start': int(start_frame),
        'frame_end': int(end_frame),
        'mean_reproj_px': float(error_px),
        'mean_reproj_mm': float(error_mm),
    }


def contiguous_high_error_segments(df, column, threshold):
    mask = df[column] >= threshold
    segments = []
    segment_start = None
    for idx, is_high in enumerate(mask):
        if is_high and segment_start is None:
            segment_start = idx
        elif not is_high and segment_start is not None:
            segments.append((segment_start, idx - 1))
            segment_start = None
    if segment_start is not None:
        segments.append((segment_start, len(df) - 1))

    rows = []
    for start_idx, end_idx in segments:
        rows.append({
            'start_time': df.iloc[start_idx]['time'],
            'end_time': df.iloc[end_idx]['time'],
            'peak_rms': df.iloc[start_idx:end_idx + 1]['marker_error_RMS'].max(),
        })
    return rows


def main():
    parser = argparse.ArgumentParser(description='Summarize Pose2Sim triangulation and IK quality metrics.')
    parser.add_argument('project_dir', nargs='?', default='.', help='Pose2Sim trial directory')
    args = parser.parse_args()

    project_dir = Path(args.project_dir).resolve()
    log_path = project_dir / 'logs.txt'
    sto_candidates = sorted((project_dir / 'kinematics').glob('*_ik_marker_errors.sto'))
    if not sto_candidates:
        sto_candidates = sorted((project_dir / 'kinematics').glob('_ik_marker_errors.sto'))
    if not sto_candidates:
        raise FileNotFoundError(f'No IK marker error file found in {project_dir / "kinematics"}')

    triang_metrics = parse_latest_triangulation_metrics(log_path) if log_path.exists() else None
    ik_df = load_ik_errors(sto_candidates[0])
    rms_p95 = ik_df['marker_error_RMS'].quantile(0.95)
    high_error_segments = contiguous_high_error_segments(ik_df, 'marker_error_RMS', rms_p95)

    print(f'Project: {project_dir.name}')
    if triang_metrics is not None:
        print(
            f"Triangulation: {triang_metrics['mean_reproj_px']:.2f} px, "
            f"{triang_metrics['mean_reproj_mm']:.1f} mm "
            f"(frames {triang_metrics['frame_start']}-{triang_metrics['frame_end']})"
        )
    else:
        print('Triangulation: not found in logs.txt')

    print(
        'IK RMS: '
        f"mean={ik_df['marker_error_RMS'].mean():.5f}, "
        f"median={ik_df['marker_error_RMS'].median():.5f}, "
        f"p95={rms_p95:.5f}, "
        f"max={ik_df['marker_error_RMS'].max():.5f}"
    )
    print(
        'IK max marker error: '
        f"mean={ik_df['marker_error_max'].mean():.5f}, "
        f"p95={ik_df['marker_error_max'].quantile(0.95):.5f}, "
        f"max={ik_df['marker_error_max'].max():.5f}"
    )
    print('Top RMS frames:')
    print(ik_df.nlargest(10, 'marker_error_RMS')[['time', 'marker_error_RMS', 'marker_error_max']].to_string(index=False))

    if high_error_segments:
        print('High-error segments (RMS >= p95):')
        print(pd.DataFrame(high_error_segments).to_string(index=False))


if __name__ == '__main__':
    main()
