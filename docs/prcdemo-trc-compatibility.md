# PRCDemo TRC Compatibility Check

## What was reproduced

The Windows PRCDemo executable cannot be run in this Linux workspace because Wine is not installed, but its Unity C# assembly was inspected from:

- `PRCDemo/PVTest_Data/Managed/Assembly-CSharp.dll`

The PRCDemo loader default axis map is:

- `SourceRightAxis = X`
- `SourceUpAxis = -Y`
- `SourceForwardAxis = Z`

So PRCDemo converts every TRC marker as `Unity = (X, -Y, Z)`.

Using that same mapping and PRCDemo's static assessment formulas, the current guarded TRC reproduces the screenshot result:

- average static score: `1.750/3`
- high/low shoulder: `0.962 cm`, score `2/3`
- head tilt: `56.320 deg`, score `0/3`
- load bias: `100.000%`, score `0/3`
- foot progression: `110.539 deg`, score `0/3`

Raw evidence:

- `data/rigid_triangulation_demo0417v6_full/prcdemo_static_assessment.csv`

## Root cause

The rigid output did not collapse the 3D hip width. The TRC has a stable anatomical hip width around `24.53 cm`.

The PRCDemo display looked narrow because this trial's anatomical left-right width is mostly in Pose2Sim `Z`, while PRCDemo treats source `X` as left-right:

| Variant | Pair | PRCDemo Unity X span | PRCDemo Unity Z span | 3D distance |
| --- | --- | ---: | ---: | ---: |
| current TRC axes | LHip-RHip | 2.43 cm | 24.47 cm | 24.59 cm |
| current TRC axes | LShoulder-RShoulder | 2.73 cm | 36.05 cm | 36.16 cm |
| PRCDemo visual axes | LHip-RHip | 24.47 cm | 2.43 cm | 24.59 cm |
| PRCDemo visual axes | LShoulder-RShoulder | 36.05 cm | 2.73 cm | 36.16 cm |

Figure:

![PRCDemo axis diagnostic](../figures/rigid_triangulation_demo0417v6_full/prcdemo_axis_diagnostic.png)

## Code changed

Added:

- `scripts/prcdemo_trc_audit.py`

The script does three things:

1. Reproduces PRCDemo-style static metrics from a TRC file.
2. Writes an axis diagnostic CSV showing which coordinate axis PRCDemo uses as left-right/depth.
3. Exports a PRCDemo visualization TRC with only a coordinate remap:

```text
X_prcdemo = Z_pose2sim
Y_prcdemo = Y_pose2sim
Z_prcdemo = -X_pose2sim
```

No marker distances are scaled or changed by this export; it is a rigid coordinate-axis conversion for PRCDemo viewing.

## Before / after output

Before, use this TRC for Pose2Sim/OpenSim-style coordinates:

- `data/rigid_triangulation_demo0417v6_full/rigid_guarded/demo0417v6_0-390_rigid_guarded.trc`

After, use this TRC in PRCDemo on Windows:

- `data/rigid_triangulation_demo0417v6_full/rigid_guarded/demo0417v6_0-390_rigid_guarded_prcdemo.trc`

The before and after 3D distances are unchanged:

| Pair | Before mean | After mean |
| --- | ---: | ---: |
| LHip-RHip | 24.527 cm | 24.527 cm |
| LShoulder-RShoulder | 36.175 cm | 36.175 cm |
| LAnkle-RAnkle | 15.598 cm | 15.598 cm |

## Limits

The PRCDemo static score is a heuristic for a controlled static posture. The remapped PRCDemo visual TRC makes left-right proportions display correctly, but PRCDemo's O/X-leg items still score low because its formulas directly threshold knee/ankle X separation. That should not be read as proof that the triangulation geometry is wrong unless the input trial was captured in PRCDemo's expected static assessment stance.
