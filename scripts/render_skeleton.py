#!/usr/bin/env python3
"""Render HALPE_26 skeleton frames from a TRC as PNG (top-down X-Z + front X-Y),
so the pelvis L/R and twist can be eyeballed. Pelvis bones red, shoulders blue,
right side solid / left side dashed, RHip & RShoulder marked 'R'.
Usage: render_skeleton.py <trc> <out.png> f1 f2 f3 ...
"""
import sys
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# 1-indexed TRC marker order
IDX = {n: i for i, n in enumerate(
    ['Hip','RHip','RKnee','RAnkle','RBigToe','RSmallToe','RHeel','LHip','LKnee','LAnkle',
     'LBigToe','LSmallToe','LHeel','Neck','Head','Nose','REye','LEye','REar','LEar',
     'RShoulder','RElbow','RWrist','LShoulder','LElbow','LWrist'], start=1)}
BONES = [('Hip','Neck','k'),('Neck','Head','k'),
         ('Hip','RHip','r'),('RHip','RKnee','dimgray'),('RKnee','RAnkle','dimgray'),
         ('Hip','LHip','salmon'),('LHip','LKnee','silver'),('LKnee','LAnkle','silver'),
         ('Neck','RShoulder','b'),('RShoulder','RElbow','steelblue'),('RElbow','RWrist','steelblue'),
         ('Neck','LShoulder','c'),('LShoulder','LElbow','lightblue'),('LElbow','LWrist','lightblue'),
         ('RAnkle','RHeel','dimgray'),('RAnkle','RBigToe','dimgray'),
         ('LAnkle','LHeel','silver'),('LAnkle','LBigToe','silver')]


def load(trc):
    rows = []
    for ln in open(trc).readlines()[5:]:
        p = ln.split('\t')
        if len(p) < 5: continue
        rows.append([float(x) if x.strip() not in ('', '\n') else np.nan for x in p])
    return np.array(rows)


def xyz(arr, i, name):
    c = 2 + (IDX[name]-1)*3
    return arr[i, c], arr[i, c+1], arr[i, c+2]


def render(trc, out, frames):
    arr = load(trc); fr = arr[:, 0].astype(int)
    n = len(frames)
    fig, axes = plt.subplots(2, n, figsize=(3.0*n, 6), squeeze=False)
    for col, f in enumerate(frames):
        i = np.where(fr == f)[0]
        if len(i) == 0:
            continue
        i = i[0]
        for view, (a, b) in enumerate([((0, 'X'), (2, 'Z(depth)')), ((0, 'X'), (1, 'Y(up)'))]):
            ax = axes[view][col]
            for m1, m2, colr in BONES:
                p1 = xyz(arr, i, m1); p2 = xyz(arr, i, m2)
                ls = '--' if (m1.startswith('L') or m2.startswith('L')) else '-'
                ax.plot([p1[a[0]], p2[a[0]]], [p1[b[0]], p2[b[0]]], ls, color=colr, lw=2)
            for mk, lab in [('RHip', 'R'), ('LHip', 'L'), ('RShoulder', 'rS'), ('LShoulder', 'lS')]:
                p = xyz(arr, i, mk)
                ax.scatter([p[a[0]]], [p[b[0]]], s=18, c='red' if mk.startswith('R') else 'green', zorder=5)
                ax.annotate(lab, (p[a[0]], p[b[0]]), fontsize=8)
            if view == 1:
                ax.invert_yaxis()  # Y more negative = up
            ax.set_aspect('equal'); ax.set_title(f'f{f} {b[1] if view else "top"}', fontsize=9)
            ax.tick_params(labelsize=6)
    fig.suptitle(trc.split('/')[-1], fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=80)
    print('WROTE', out)


if __name__ == '__main__':
    render(sys.argv[1], sys.argv[2], [int(x) for x in sys.argv[3:]])
