#!/usr/bin/env python3
"""Measure residual pelvis jitter/flip in a TRC.

The chirality guard stops the rigid pelvis from writing a 180-deg L/R swap, but on
rejected frames it falls back to the *independent* triangulation, which under the same
foreshortening is degenerate: the pelvis lateral width collapses (22 cm -> 9-16 cm) and
wobbles -> the "hips shaking / flipping" the user sees in PRCDemo.

Metrics per frame (horizontal X-Z plane):
  pelW   = |RHip - LHip|            (rigid pelvis ~ constant 22 cm; a drop = collapse)
  pel.sh = (RHip-LHip).(RSh-LSh)    (chirality; < 0 = visible L/R swap)

A frame is COLLAPSE if pelW < collapse_cm, SWAP if pel.sh < 0.
Usage: verify_pelvis_jitter.py <trc> [lo hi]
"""
import sys
import numpy as np

MARK = {'Hip': 1, 'RHip': 2, 'LHip': 8, 'RShoulder': 21, 'LShoulder': 24}


def load(trc):
    with open(trc) as f:
        lines = f.readlines()
    arr = []
    for ln in lines[5:]:
        p = ln.split('\t')
        if len(p) < 5:
            continue
        arr.append([float(x) if x.strip() not in ('', '\n') else np.nan for x in p])
    return np.array(arr)


def col(m):
    return 2 + (m - 1) * 3


def metrics(arr):
    g = {k: arr[:, col(v):col(v) + 3] for k, v in MARK.items()}
    pel = g['RHip'] - g['LHip']
    sh = g['RShoulder'] - g['LShoulder']
    pdot = pel[:, 0] * sh[:, 0] + pel[:, 2] * sh[:, 2]
    pw = np.sqrt(pel[:, 0] ** 2 + pel[:, 2] ** 2) * 100.0
    return arr[:, 0].astype(int), pw, pdot


def report(trc, lo=None, hi=None, collapse_cm=18.0):
    arr = load(trc)
    frame, pw, pdot = metrics(arr)
    n = len(frame)
    lo = 0 if lo is None else lo
    hi = n if hi is None else hi
    sel = (np.arange(n) >= lo) & (np.arange(n) < hi)
    finite = np.isfinite(pw) & sel
    collapse = finite & (pw < collapse_cm)
    swap = finite & (pdot < 0)
    print(f'{trc}')
    print(f'  window [{lo},{hi})  median pelW = {np.nanmedian(pw[finite]):.1f} cm')
    print(f'  COLLAPSE (pelW<{collapse_cm:.0f}cm): {int(collapse.sum())} frames -> {list(frame[collapse])}')
    print(f'  SWAP (pel.sh<0):                {int(swap.sum())} frames -> {list(frame[swap])}')
    return int(collapse.sum()), int(swap.sum())


if __name__ == '__main__':
    trc = sys.argv[1]
    lo = int(sys.argv[2]) if len(sys.argv) > 2 else None
    hi = int(sys.argv[3]) if len(sys.argv) > 3 else None
    report(trc, lo, hi)
