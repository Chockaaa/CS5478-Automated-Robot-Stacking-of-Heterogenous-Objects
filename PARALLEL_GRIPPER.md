# Parallel-gripper stacking: tested simulation

The branch `feature/parallel-gripper-stacking` now runs on native Windows
in the Conda environment `ravens-stacking` (Python 3.11).
The original TensorFlow training dependencies are not installed in this environment.

## Run

From the repository root in Anaconda PowerShell:

```powershell
conda activate ravens-stacking
python -m ravens.stacking_smoke --task=stack-block-pyramid-parallel --episodes=1 --assisted --disp
```

Different-sized boxes:

```powershell
python -m ravens.stacking_smoke --task=stack-box-tower-parallel --episodes=1 --assisted --disp
```

Boxes and an upright cylinder:

```powershell
python -m ravens.stacking_smoke --task=stack-mixed-tower-parallel --episodes=1 --assisted --disp
```

Omit `--disp` for headless execution. Omit `--assisted` to test friction alone.
Use `--seed=3` to reproduce the failing assisted mixed-object scene.
Use `--episodes=5 --output=my-results.json --snapshots` to save a batch of
results and final camera images. The script returns a nonzero exit code if any
trial fails. Some Conda versions on Windows mask that exit code; inspect the
JSON results as well.

## Recreate the environment

The environment on this computer was cloned from the existing `stacking`
environment to reuse its working Conda PyBullet build, then simulation
dependencies were installed. The source environment was not modified.
This minimal recipe is provided for a fresh installation; it has not been
independently recreated from scratch:

```powershell
conda env create -f environment-stacking.yml
conda activate ravens-stacking
python -m unittest ravens.tasks.parallel_gripper_test -v
```

No editable installation is required when running from the repository root.
Do not run `pip install -r requirements.txt` in this simulation environment:
that file belongs to the old TensorFlow training setup.

## What is implemented

- A simple parallel gripper with two motor-driven sliding fingers, a 90 mm
  opening, physical collision shapes, and friction. This replaces the earlier
  unverified Robotiq integration; it is not a calibrated Robotiq 2F-85 model.
- Both fingers must contact the same movable object before grasp acceptance.
  Empty and off-centre grasps cannot activate assistance.
- Optional assistance adds a fixed constraint only after bilateral contact.
  It is removed before opening the fingers. Released stacks remain physical.
- A scripted expert chooses object-centre picks and object target poses using
  exact simulator state. The controller measures the object-to-tool transform
  after lifting to compensate for movement during closure.
- The six-cube pyramid uses a 3-2-1 layout. Initial cubes have jaw clearance;
  their position jitter, their yaw, and the base yaw vary with the seed.
- Box towers use three square-footprint boxes (60, 48, 36 mm wide), random
  heights between 32 and 45 mm, random initial yaw, and a random tower location.
  The mixed task replaces the middle box with an upright cylinder.
- Layers are built bottom-up. Tower height is computed from actual dimensions.
  Motion and settling waits are bounded.

This tests a scripted stacking controller, not visual grasp learning.
The action convention now uses object-centre heights and absolute object target
rotations. Stock Transporter uses surface-height picks and a different rotation
convention, so the old training commands are not compatible as-is. An action
adapter, grasp-orientation output, and compatible demonstration collection are
needed before training. No neural-network training was run.

## Validation

Native Windows results, seeds 0 through 4:

| Task | Contact-assisted | Friction only |
| --- | --- | --- |
| Six-cube pyramid | 5/5 | 2/5 |
| Different-sized box tower | 5/5 | 0/5 |
| Box-cylinder-box tower | 4/5 | 1/5 |

The three independent contact/release unit tests pass, and
`python -m pip check` reports no broken requirements in the created environment.

An additional original suction-pyramid regression reaches full Ravens reward.
Its strict XYZ check fails as expected because the original task's target
heights include air gaps (up to 27.5 mm above the settled cube centres);
see `validation/suction-regression.json`. The new parallel task corrects those
target heights.

See the six `validation/*-assisted.json` and `*-friction.json` files for seeded
trial results. Exploratory results from earlier code revisions are separately
stored in `validation/exploratory/`; do not combine them with the final results.

A success requires full task reward and, after 480 additional simulation steps:
a unique matching of every object to a target within 10 mm in XYZ, upright
orientation (local up within approximately 11.5 degrees of vertical), and
load-bearing upward contact for each object. This prevents the original XY-only
Ravens reward from reporting a tabletop arrangement as a stack.

Five seeds are a small smoke test, not a benchmark guarantee. Assisted box
stacking is the usable baseline; friction-only grasping still slips or tilts.
One assisted mixed-object scene ejects the cylinder and hits the settling limit.
Irregular meshes, spheres, rectangular-footprint grasp planning, general clutter,
continuous-control tasks, and real robot execution were not validated.
