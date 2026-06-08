#!/usr/bin/env python3
"""Sweep a TRC for ALGORITHM-LEVEL pelvis artifacts (not data/occlusion limits).

Classes (all should be ~0 for a clean rigid pelvis):
  bone_drift  : 3D |RHip-LHip| deviates >2cm from the trial median (rigid => constant;
                a drift means an independent-fallback / degenerate frame leaked through)
  lr_swap     : pelvis lateral axis points opposite the shoulders (>90deg), shoulders clear
  nonanat_tw  : trunk axial twist |pelvis-vs-shoulder yaw - offset| > tw_limit deg
                (anatomical trunk rotation maxes ~45deg, so >50 is non-physical)
  teleport    : Hip 3D jumps > jump_cm vs BOTH neighbours (single-frame pop)
Usage: sweep_artifacts.py <trc> [tw_limit=50] [jump_cm=12]
"""
import sys, glob
import numpy as np
sys.path.insert(0, 'scripts')
from verify_pelvis_jitter import load, col, MARK

M = dict(MARK); M.update({'Neck': 14, 'Hip': 1})


def runs(frames):
    if len(frames) == 0:
        return '-'
    frames = sorted(int(x) for x in frames); seg = []; s = p = frames[0]
    for v in frames[1:]:
        if v == p + 1: p = v
        else: seg.append((s, p)); s = p = v
    seg.append((s, p))
    return ','.join(f'{a}-{b}' if a != b else f'{a}' for a, b in seg)


def sweep(trc, tw_limit=50.0, jump_cm=12.0):
    arr = load(trc); g = {k: arr[:, col(v):col(v) + 3] for k, v in M.items()}
    f = arr[:, 0].astype(int)
    pel = g['RHip'] - g['LHip']; sh = g['RShoulder'] - g['LShoulder']
    d3 = np.linalg.norm(pel, axis=1) * 100
    med3 = np.nanmedian(d3)
    bone = np.isfinite(d3) & (np.abs(d3 - med3) > 2.0)
    shw = np.sqrt((sh[:, [0, 2]] ** 2).sum(1)) * 100
    clear = shw > 30
    # twist about torso-up
    up = g['Neck'] - g['Hip']; upn = up / (np.linalg.norm(up, axis=1, keepdims=True) + 1e-9)
    def perp(v): return v - np.sum(v * upn, axis=1, keepdims=True) * upn
    pp = perp(pel); ss = perp(sh)
    s = np.sum(np.cross(ss, pp) * upn, axis=1); c = np.sum(ss * pp, axis=1)
    tw = np.degrees(np.arctan2(s, c))
    valid = np.isfinite(tw) & clear
    off = np.median(tw[valid]) if valid.any() else 0.0
    rel = tw - off
    swap = valid & (np.abs(rel) > 90)
    nonanat = valid & (np.abs(rel) > tw_limit) & (~swap)
    # teleport: hip jump vs both neighbours
    hip = g['Hip']
    jmp = np.full(len(f), 0.0)
    for i in range(1, len(f) - 1):
        a = np.linalg.norm(hip[i] - hip[i - 1]) * 100
        b = np.linalg.norm(hip[i] - hip[i + 1]) * 100
        if np.isfinite(a) and np.isfinite(b):
            jmp[i] = min(a, b)
    teleport = jmp > jump_cm
    return dict(n=len(f), med_width=med3,
                bone=f[bone], lr_swap=f[swap], nonanat=f[nonanat], teleport=f[teleport],
                tw_p1=np.percentile(rel[valid], 1) if valid.any() else 0,
                tw_p99=np.percentile(rel[valid], 99) if valid.any() else 0)


if __name__ == '__main__':
    trc = sys.argv[1]
    tw = float(sys.argv[2]) if len(sys.argv) > 2 else 50.0
    jc = float(sys.argv[3]) if len(sys.argv) > 3 else 12.0
    r = sweep(trc, tw, jc)
    name = trc.split('/')[-1][:40]
    print(f'{name:42s} n={r["n"]:4d} W={r["med_width"]:.1f}cm tw[p1,p99]=[{r["tw_p1"]:+.0f},{r["tw_p99"]:+.0f}]')
    print(f'   bone_drift[{len(r["bone"])}]={runs(r["bone"])[:80]}')
    print(f'   lr_swap[{len(r["lr_swap"])}]={runs(r["lr_swap"])[:80]}')
    print(f'   nonanat_twist>{int(tw)}[{len(r["nonanat"])}]={runs(r["nonanat"])[:80]}')
    print(f'   teleport[{len(r["teleport"])}]={runs(r["teleport"])[:80]}')
