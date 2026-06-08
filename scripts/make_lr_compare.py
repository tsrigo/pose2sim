#!/usr/bin/env python3
"""Visual before/after evidence for the pelvis L/R-swap fix.

BEFORE = rigid groups with the pelvis chirality guard DISABLED (chirality_guard=false).
AFTER  = same pipeline with the guard ON (production). Same cached 2D pose, so the ONLY
difference is the guard -> any change is attributable to the fix.

Outputs (into <outdir>):
  lr_metric_timeline_<subj>.png  - signed (RHip-LHip).body_left over frames; <0 correct, >0 swapped
  lr_pelvis_frame_<subj>.png     - top-down pelvis+shoulder bones at a representative swap frame
  zf_skeleton_before_after.mp4   - side-by-side 3D skeleton over zf's swap window
"""
import sys, glob
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib import animation

OUT = Path(sys.argv[1]) if len(sys.argv) > 1 else Path('/tmp/lr_compare')
OUT.mkdir(parents=True, exist_ok=True)

PROD = Path('data/exp_lr_swap/prod_20260602')
# subject -> (before noguard trc, after guard trc)
NG = {p.name: f'/tmp/ng_{p.name[:4]}.trc' for p in sorted(PROD.iterdir()) if p.is_dir()}


def load(p):
    L = open(p).read().splitlines()
    names = [n for n in L[3].split('\t')[2:] if n.strip()]
    idx = {n: i for i, n in enumerate(names)}
    rows = []
    for ln in L[5:]:
        if not ln.strip() or ln.split('\t')[0].strip() in ('', 'X1'):
            continue
        rows.append([float(x) if x.strip() not in ('', 'nan', 'NaN') else np.nan for x in ln.split('\t')])
    a = np.array(rows)
    return {n: a[:, 2 + idx[n] * 3:2 + idx[n] * 3 + 3] for n in names}


def body_left(d):
    bl = d['LShoulder'] - d['RShoulder']
    bl = bl / (np.linalg.norm(bl, axis=1, keepdims=True) + 1e-9)
    return bl


def signed_lat(d):
    """ (RHip-LHip) . body_left ; correct labeling -> negative (R on body's right). """
    return np.nansum((d['RHip'] - d['LHip']) * body_left(d), axis=1)


subjects = list(NG)
for subj in subjects:
    lbl = subj.split('_')[0]
    after = load(glob.glob(str(PROD / subj / 'pose-3d' / '*_0-*.trc'))[0])
    before = load(NG[subj])
    n = min(len(after['RHip']), len(before['RHip']))
    sb = signed_lat(before)[:n]
    sa = signed_lat(after)[:n]
    swap = sb > 0  # before is swapped where metric goes positive

    # --- timeline figure ---
    fig, ax = plt.subplots(figsize=(12, 3.6))
    ax.axhline(0, color='k', lw=0.8)
    ax.fill_between(range(n), 0, 1, where=swap, transform=ax.get_xaxis_transform(),
                    color='red', alpha=0.10, label='BEFORE swapped region')
    ax.plot(sb, color='#d62728', lw=1.1, label='BEFORE (guard off)')
    ax.plot(sa, color='#2ca02c', lw=1.1, label='AFTER (guard on)')
    ax.set_title(f'{lbl} pelvis left/right indicator   (RHip−LHip)·body-left   '
                 f'[<0 = correct,  >0 = L/R reversed]', fontsize=10)
    ax.set_xlabel('frame'); ax.set_ylabel('signed lateral (m)')
    ax.text(0.005, 0.04, f'BEFORE reversed frames: {int(swap.sum())}   AFTER reversed frames: '
            f'{int((sa>0).sum())}', transform=ax.transAxes, fontsize=9,
            bbox=dict(fc='white', ec='gray', alpha=0.8))
    ax.legend(loc='upper right', fontsize=8)
    ax.set_ylim(min(-0.25, np.nanmin(sa) - 0.02), max(0.25, np.nanmax(sb) + 0.02))
    fig.tight_layout()
    fig.savefig(OUT / f'lr_metric_timeline_{subj[:6]}.png', dpi=130)
    plt.close(fig)

    # --- representative swap frame: top-down pelvis + shoulders ---
    if swap.any():
        f0 = int(np.flatnonzero(swap)[len(np.flatnonzero(swap)) // 2])
        fig, axs = plt.subplots(1, 2, figsize=(9, 4.4))
        for ax, d, ttl in [(axs[0], before, 'BEFORE (guard off)'),
                           (axs[1], after, 'AFTER (guard on)')]:
            def XZ(m):
                return d[m][f0][0], d[m][f0][2]
            hip = XZ('Hip')
            for m, c in [('RHip', '#d62728'), ('LHip', '#1f77b4')]:
                x, z = XZ(m)
                ax.plot([hip[0], x], [hip[1], z], '-', color=c, lw=3)
                ax.scatter([x], [z], color=c, s=70, zorder=5)
                ax.annotate(m, (x, z), color=c, fontsize=9)
            for m, c in [('RShoulder', '#ff9896'), ('LShoulder', '#aec7e8')]:
                x, z = XZ(m)
                ax.scatter([x], [z], color=c, s=60, marker='s', zorder=5)
                ax.annotate(m, (x, z), color=c, fontsize=8)
            ax.scatter([hip[0]], [hip[1]], color='k', s=40, zorder=6)
            ax.set_title(ttl, fontsize=10); ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)')
            ax.set_aspect('equal'); ax.grid(alpha=0.3)
        fig.suptitle(f'{lbl} frame {f0}, top-down (red=Right, blue=Left). '
                     'BEFORE: RHip on the wrong side.', fontsize=10)
        fig.tight_layout()
        fig.savefig(OUT / f'lr_pelvis_frame_{subj[:6]}.png', dpi=130)
        plt.close(fig)
    print(f'{subj}: before swap frames={int(swap.sum())}, after={int((sa>0).sum())}')

# ---------- zf side-by-side 3D skeleton video over the swap window ----------
zf = [s for s in subjects if s.startswith('zf')][0]
after = load(glob.glob(str(PROD / zf / 'pose-3d' / '*_0-*.trc'))[0])
before = load(NG[zf])
n = min(len(after['RHip']), len(before['RHip']))
swap = signed_lat(before)[:n] > 0
sf = np.flatnonzero(swap)
lo, hi = max(0, sf.min() - 12), min(n, sf.max() + 12)
frames = list(range(lo, hi))

BONES = [('Hip', 'RHip'), ('Hip', 'LHip'), ('RHip', 'RKnee'), ('RKnee', 'RAnkle'),
         ('LHip', 'LKnee'), ('LKnee', 'LAnkle'), ('Hip', 'Neck'),
         ('Neck', 'RShoulder'), ('Neck', 'LShoulder')]
RIGHT = {'RHip', 'RKnee', 'RAnkle', 'RShoulder'}
LEFT = {'LHip', 'LKnee', 'LAnkle', 'LShoulder'}
MARKERS = ['Hip', 'Neck', 'RHip', 'LHip', 'RKnee', 'LKnee', 'RAnkle', 'LAnkle', 'RShoulder', 'LShoulder']


RAD = 0.55  # half-window (m) around the hip — top-down world view
fig, (ax_b, ax_a) = plt.subplots(1, 2, figsize=(11, 5.8))


def col(m):
    return '#d62728' if m in RIGHT else ('#1f77b4' if m in LEFT else 'k')


def draw(ax, d, f, title):
    ax.clear()
    hip = d['Hip'][f]
    cx, cz = (hip[0], hip[2]) if np.all(np.isfinite(hip)) else (1.3, 0.0)
    # body-left direction (from shoulders) as an arrow
    bl = d['LShoulder'][f] - d['RShoulder'][f]
    if np.all(np.isfinite(bl)):
        u = bl[[0, 2]] / (np.linalg.norm(bl[[0, 2]]) + 1e-9) * 0.3
        ax.annotate('', xy=(cx + u[0], cz + u[1]), xytext=(cx, cz),
                    arrowprops=dict(arrowstyle='-|>', color='gray', lw=1.5))
        ax.text(cx + u[0] * 1.05, cz + u[1] * 1.05, "body's LEFT", color='gray', fontsize=8)
    for a, b in BONES:
        pa, pb = d[a][f], d[b][f]
        if np.all(np.isfinite(pa)) and np.all(np.isfinite(pb)):
            c = col(b) if b in RIGHT | LEFT else col(a)
            ax.plot([pa[0], pb[0]], [pa[2], pb[2]], color=c, lw=3)
    for m in MARKERS:
        p = d[m][f]
        if np.all(np.isfinite(p)):
            ax.scatter([p[0]], [p[2]], color=col(m), s=70, zorder=5)
    for m in ['RHip', 'LHip']:
        p = d[m][f]
        if np.all(np.isfinite(p)):
            ax.annotate(m, (p[0], p[2]), color=col(m), fontsize=9, zorder=6)
    ax.set_xlim(cx - RAD, cx + RAD); ax.set_ylim(cz - RAD, cz + RAD)
    ax.set_aspect('equal'); ax.grid(alpha=0.3)
    ax.set_xlabel('X (m)'); ax.set_ylabel('Z (m)'); ax.set_title(title, fontsize=11)


def update(i):
    f = frames[i]
    draw(ax_b, before, f, f'BEFORE — guard off  (frame {f})')
    draw(ax_a, after, f, f'AFTER — guard on  (frame {f})')
    fig.suptitle('Top-down world view.  Red = Right side, Blue = Left side.  '
                 'BEFORE: the right hip/leg flips across to the left during the turn.', fontsize=11)
    return []


anim = animation.FuncAnimation(fig, update, frames=len(frames), interval=100)
vid = OUT / 'zf_skeleton_before_after.mp4'
anim.save(str(vid), writer=animation.FFMpegWriter(fps=10, bitrate=2400))
plt.close(fig)
print(f'wrote {vid} ({len(frames)} frames, window {lo}-{hi})')
print(f'\nAll comparison assets in {OUT}')
