# Ravens - Transporter Networks

## Project extension: heterogeneous object stacking

Our goal is to pick randomly placed rigid objects of different shapes and sizes
with a parallel gripper and build a stable tower. We will start with Transporter
Networks, then compare search-based and language-guided approaches using the
same manipulation and end-to-end evaluation infrastructure.

### Current implementation versus planned work

- **Implemented:** `stack-robotiq`, a six-cube 3-2-1 pyramid task with aligned
  40 mm cubes, a visible Robotiq gripper, and contact-based grasping.
- **Implemented:** `transporter-robotiq`, using the repository's Attention and
  Transport networks to predict actions from RGB-D observations.
- **Validated:** the scripted oracle completed five tested scenes; neural
  training, checkpoint loading, and action execution passed smoke checks.
- **Pyramid checkpoint limitation:** its original one-update checkpoint is only
  an integration test, not a trained stacking policy.
- **Mixed-object pipeline implemented:** `stack-heterogeneous` generates one cube,
  one cuboid, and one upright cylinder with seeded dimensions, positions, and
  yaw. It includes a physical oracle, contact-based grasp execution, tower
  scoring, demonstration collection, and a Transporter training/testing runner.
- **Policy scope:** `transporter-heterogeneous` learns pick/place XY with the
  original Attention and Transport ResNets. A separate RGB-D geometry adapter
  refines the selected grasp centre and supplies yaw and height. This is not learned rotation prediction or
  end-to-end learned motor control; evaluation covers the complete hybrid system.
- **Still planned:** learned grasp rotations, search planners, language-guided
  policies, arbitrary 3D reorientation, and broader shape generalization.

For the runnable gripper baseline, use [the Robotiq guide](docs/robotiq.md)
and [its Python 3.10 Conda environment](environment-robotiq.yml), not the legacy
Python 3.7 installation below. From the repository root:

```shell
conda env create -f environment-robotiq.yml  # Skip if already created.
conda activate ravens-transporter
python -m unittest ravens.tasks.robotiq_test -v
python -m ravens.robotiq_run --mode=oracle --episodes=1 --disp
```

`--mode=oracle` is scripted demonstration generation, not neural inference.
`--mode=policy` loads Transporter weights; see the guide for collection,
training, and evaluation commands. Existing suction checkpoints are not
interchangeable with the gripper policy.

### Step 1: preview a mixed-object scene

With `ravens-transporter` activated, run:

```shell
python -m ravens.heterogeneous_preview --seed=0 --split=train
python -m unittest ravens.tasks.stack_heterogeneous_test ravens.tasks.robotiq_test -v
```

The preview saves `train-0.png` and a versioned `train-0.json` scene manifest in
`robotiq-results/heterogeneous/`. No robot actions are executed. The manifest
records full object dimensions in metres, initial poses, mass, friction, colors,
and the intended tower region. Use another non-negative seed or choose
`--split=validation` / `--split=test` for different scenes. Repeating a seed and
split reproduces the manifest independently of NumPy's global random state.
These are separate random streams from the same object distribution, not yet
held-out-shape or physically verified-stackable datasets.

Scene version 2 adds a camera-visible cyan target square (visual only, no
collision or support fixture) and settles objects before the first observation.
It uses the contact-hold gripper controller. Version 1 demonstrations and models
are retained separately; do not mix them with the version 2 baseline.

Each scene contains exactly three objects: a cube with 32–50 mm sides, a cuboid
with 45–60 × 28–40 × 30–55 mm extents, and a cylinder with 32–50 mm diameter and
30–55 mm height. All begin upright and separated. This is not yet a benchmark
of arbitrary household objects or arbitrary 3D poses.

Scene tests cover 300 sampled configurations, reproducibility, invalid inputs,
and a real PyBullet environment reset. Further tests check tower/collapse scoring,
gripper motion, and observation-only action decoding.

### Mixed-object training and end-to-end testing

Run from the repository root in `ravens-transporter`. GPU setup for the tested
Windows machine is documented in [the environment guide](docs/robotiq.md#environment).
Use these dedicated mixed-object runners; the upstream commands later in this
README describe the original Ravens tasks, not this benchmark protocol.

```shell
# Collect 30 NEW successful demos; every attempted episode is logged.
python -m ravens.heterogeneous_run --mode=demos --split=train --episodes=30 --attempts=70 --output=robotiq-results/my-mixed-collection

# Train both Transporter networks from scratch on the collected train split.
python -m ravens.heterogeneous_train --steps=1000 --save_every=200 --compile_models

# Use validation for development; reserve test for a frozen policy.
python -m ravens.heterogeneous_run --mode=policy --split=validation --episodes=5 --checkpoint=1000 --output=robotiq-results/my-mixed-validation
python -m ravens.heterogeneous_run --mode=policy --split=test --seed=100 --episodes=10 --checkpoint=1000 --output=robotiq-results/my-mixed-test

# Oracle reference on the SAME test scenes, not a neural-policy result.
python -m ravens.heterogeneous_run --mode=oracle --split=test --seed=100 --episodes=10 --output=robotiq-results/my-mixed-oracle-test

python -m unittest ravens.tasks.stack_heterogeneous_test ravens.tasks.robotiq_test ravens.agents.transporter_heterogeneous_test ravens.heterogeneous_run_test ravens.heterogeneous_train_test -v
```

The default model name is `mixed-tower-v2`, and the default dataset root is
`robotiq-dataset/mixed-v2`; use the same `--name`, `--data_dir`, and
`--train_dir` when training and loading. `--resume=N` reloads weights from step N
but starts fresh optimizer state. Checkpoints, datasets, and run artifacts are
local and Git-ignored. Existing checkpoints/results are protected against
accidental overwrite: choose a new model name/output directory for each experiment.
If the local checkpoint already exists, skip collection/training to replay it;
use a new `--name` to train another model from scratch.
`--compile_models` traces the same Keras networks with `tf.function` to reduce
eager execution overhead. It does not change the architecture; small floating
point differences are possible. Omit it to use the original eager execution.

The baseline uses a 192x112 RGB-D heightmap at 4 mm/pixel, crop size 32, and one
rotation bin. Attention selects the object; a 1 mm RGB-D raster with local
sampling-hole handling refines its grasp centre. Transport selects placement XY
without a hard-coded target override. No oracle coordinates enter inference.
Attention logits are masked to the executor's 25–65 mm table-grasp height band
using the observed heightmap. This excludes empty table pixels and tall stacks;
it does not impose a scripted object order. If no feasible-height pixel exists,
the policy stops without claiming success.
The RGB-D adapter aligns jaws to the estimated short face axis;
placement preserves that grasp orientation. The controller opens fully and
closes by physical contact, holding its opening after bilateral pad contact,
with height-dependent TCP offsets. Gripper width is
not a learned output. Objects are never attached to the gripper or snapped into
position. Current scope is separated, initially upright objects on the table,
not arbitrary clutter or recovery by picking objects already high in a tower.
Training uses RGB channel/gain augmentation and excludes demonstration actions
that did not increase the stable stack count. Spatial augmentation is disabled
because the tower destination is fixed. Older demonstrations without progress
metadata can be paired with their collection `results.json` using
`--demonstration_report=PATH` on the training command. Labels are checked against
the image bounds before training starts.

Each evaluation allows six actions and a 180-second limit checked between
actions. Success requires a centred, contacting, near-upright, low-velocity
three-object support chain that passes a two-second final hold test. Cube face
symmetries are accepted. The geometric thresholds are 12 mm horizontal error,
4 mm support-height error, 10 degrees tilt, 0.01 m/s linear speed, and 0.1 rad/s
angular speed. Scoring uses simulator state; learned inference receives RGB-D
only. `results.json` records all episodes/actions/errors, `summary.json` reports
completion, partial stacking, timing, grasps, collapses, and a Wilson 95% interval.
Scene manifests and final screenshots are saved alongside them. The current
runner does not record full videos or guarantee interruption mid-inference.

Ten test scenes are a preliminary experiment, not the planned 100-trial benchmark.
Successful-demo filtering also biases the training set toward easier scenes;
evaluation must include every requested scene, not only oracle successes.

### Measured baseline (2026-10-10)

Training and held-out testing completed in the `ravens-transporter` Conda
environment on branch `feature/parallel-gripper-stacking`. This is a working
simulation baseline, **not yet reliable general-purpose stacking**.

| Frozen system, test seeds 100-109 | Complete towers | Wilson 95% interval | Mean verified objects / 3 | Contact grasps | Mean episode time |
| --- | --- | --- | --- | --- | --- |
| Transporter XY + RGB-D grasp adapter, checkpoint 1000 | 6/10 (60%) | 31.3-83.2% | 2.0 | 32/41 (78.0%) | 49.2 s |
| Ground-truth oracle, same physical gripper/controller | 9/10 (90%) | 59.6-98.2% | 2.7 | 33/33 (100%) | 19.9 s |

All ten requested scenes count, including errors. Transporter had three
scene-settling errors and one six-action failure ending with two stable objects;
the oracle had one settling error. An error receives zero **verified** stable
objects because its final hold was not completed, not because every object was
necessarily on the table. The runner completes the remaining episodes, saves
the reports, then exits with code 1 if any episode raised an error.
Times are local simulation wall times excluding model loading, not real-robot
cycle times. The collapse counter only covers decreases between completed
actions and must not be interpreted as proof that no physical falls occurred.

The model used 30 successful demonstrations from 50 attempted training scenes:
90 progress-making actions out of 122 recorded actions. Training reached step
400 in eager mode, resumed to 1000 with compiled networks, then resumed to 3000;
each resume reset Adam optimizer state. With the final decoder, checkpoint 1000
completed 2/5 validation scenes versus 1/5 for checkpoint 3000, so 1000 was
selected **before** testing. Demonstrations used contact preload 0.01; validation
selected preload 0.02 for the final controller. No test-based retuning was done.

The [compact evaluation record](validation/heterogeneous-v2.json) contains
per-seed outcomes, configurations, checkpoint/source hashes, and local raw-report
paths. Weights, datasets, screenshots and full traces are Git-ignored local
artifacts, not included in a fresh clone. All 15 targeted tests pass; `pip check`
reports no broken requirements. The original pyramid oracle also passed a
separate regression scene with the final gripper controller.

To replay the trained local model with the gripper visible (choose a fresh output
directory):

```shell
conda activate ravens-transporter
python -m ravens.heterogeneous_run --mode=policy --split=test --seed=101 --episodes=1 --checkpoint=1000 --disp --output=robotiq-results/my-replay
```

Reported evaluations were headless; this command opens the interactive viewer.
The next milestone is better grasp/placement and settling robustness, followed
by a fresh, larger held-out evaluation. CEM, beam search and VLM integrations
remain planned, not implemented.

### Active work: reliable manipulation

We are now investigating manipulation reliability, keeping the trained
Transporter weights and its RGB-D decoder unchanged. The original controller
remains the default (`--controller=baseline`); the 6/10 held-out result above is
the historical baseline, not a result for the experimental controllers.

The first diagnostic reproduced validation seed 4's settling failure. The
cylinder slipped during transfer after passing the initial lift check, then
rolled outside the workspace while the two-box stack remained standing. The
trace also showed pad bottoms below the table surface during short-object
grasps. This identifies a retention/clearance problem to investigate; it does
not establish that all failures have the same cause. Increasing the settling
timeout alone would not fix the lost grasp.

The runner now saves post-action physics diagnostics and, on exceptions, body
poses, velocities, contacts and a failure screenshot. These are privileged
diagnostics only: they do not enter learned inference. A separate development
tool traces motion, closing and release stages with the same post-action
settling as the evaluation runner:

```shell
python -m ravens.heterogeneous_diagnose --seed=4 --controller=baseline --output=robotiq-results/my-motion-trace.json
```

This tool uses the **oracle on the validation split**; it is not a policy
evaluation or a final-hold success benchmark. Use a new output file. An expected
reproduced settling failure exits with code 1 but still saves the trace.

Opt-in ablations are available through `--controller`: `clearance-v1` uses a
shallower grasp, `guarded-v1` adds a short lift and retention checks before
transfer/placement, `retention-v1` raises contact preload from 0.02 to 0.03,
`slow-v1` halves the joint-motion step size, and `cartesian-v1` adds at most
10 mm Cartesian waypoints during loaded lift, transfer and lowering while
preserving grasp yaw. The latter completed only 1/5 oracle validation towers
(seeds 0-4), versus the recorded baseline's 4/5, and was not promoted. It also
logs contact retention after transfer and before release; these contact checks
are diagnostic, not continuous slip detection. These are **experimental, not
recommended replacements**. See the
[reliability development report](validation/reliability-development.json) for
the five-scene oracle comparisons, including unsuccessful candidates. Neither
the learned weights nor the benchmark's success thresholds were changed.

Remaining work, in order:

1. Run the controlled grasp-retention suite below across each family's size/yaw
   range; measure closing contact, lift/transfer retention and pad clearance.
2. Improve grasp geometry and transport trajectories using that suite; verify
   placements without disturbing existing supports. Retention checks alone
   do not prevent a slip between checkpoints.
3. Add bounded, observation-based recovery and explicit failure reasons. Every
   retry must count toward the action/time budget; no pose snapping, oracle
   fallback, or relaxed scoring. Full fallen-object recovery is not implemented.
4. Compare promising controllers on validation seeds 5-24 before promoting one.
5. With a validated controller, collect a separate expanded demonstration set,
   retrain Transporter under a new model name, and select on validation only.
6. Freeze controller, decoder and checkpoint; run both policy and oracle on
   100 fresh test scenes (reserved seeds 1000-1099), preserving all failures.

The 100-scene held-out evaluation has **not** been completed. In parallel with
the remaining manipulation work, the expanded-data experiment below tests
whether additional demonstrations help the fixed baseline. The controller
screening round is diagnostic work, not evidence of an improved policy.
Test scenes `(test, 100-109)` stay recorded as the old baseline and are not used
for tuning this stage. Training seeds with the same integer use a different
random stream and generate different scenes.

### Single-object retention stage

Ravens remains the simulation base. `ravens.retention_benchmark` isolates one
physical object with the existing Robotiq controller; it does not train or call
Transporter. Ground-truth centered grasps deliberately remove perception error
from this diagnostic. The 108-case development matrix covers three families,
three size presets per family, three grasp yaws, two pick positions and two
transfer distances. Width and height vary together in these size presets;
independently varied dimensions belong in the later randomized confirmation.
Density and friction match the mixed-tower task, not a real-world material sweep.

```shell
conda activate ravens-transporter
python -m ravens.retention_benchmark --limit=9 --output=robotiq-results/my-retention-smoke
python -m ravens.retention_benchmark --output=robotiq-results/my-retention-matrix
python -m unittest ravens.retention_benchmark_test -v
```

Use new output directories. Each case saves contact forces, joint angle,
pad clearance and object pose relative to the gripper at 20 Hz, plus stage
boundaries. The protocol inserts a short lift and stationary holds, so its
success rate is **not comparable to full-tower success or the unmodified
production primitive**. It measures retention before release, not placement.
After the short-lift reference, every sampled checkpoint must retain bilateral
contact, with at most 5 mm translation and 10 degrees rotation. Sampling cannot
exclude slips between observations; the metric is stricter than an after-lift
contact check. Exceptions and timeouts remain failures. Exit code zero means
the diagnostic completed, not that the retention gate passed; inspect
`results.json` and `summary.development_gate_passed`.

The [stage plan](validation/retention-stage-plan.json), declared before this
benchmark ran, defines these engineering targets:

1. Complete the development matrix with >=95% retention overall and >=90% in
   each family. Deterministic-grid intervals are descriptive, not independent
   random-sample generalization evidence.
2. Freeze a promising controller and confirm those targets on 300 new randomized
   cases (100 per family), reporting uncertainty. That confirmation runner and
   experiment are future work; do not reuse development cases as held-out data.
3. Compare it against baseline with fixed Transporter weights on paired stacking
   validation scenes. Retrain only after physical execution improves.
4. Freeze the final system and achieve >=90/100 complete towers on reserved test
   seeds 1000-1099 under unchanged scoring, action and time budgets.

These are proposed simulation stage-clearance targets, not achieved results or
real-robot guarantees. No held-out test scene is used to tune the controller.

The first full baseline diagnostic completed **32/108 (29.6%)** strict-retention
cases: cubes 10/36, cuboids 15/36, cylinders 7/36. There were no exceptions or
timeouts. The largest first-failure category was the higher lift (34 of 76
failures), not horizontal transfer alone. This is not evidence that historical
tower performance changed: the protocol and metric are different. See the
[retention results](validation/retention-isolation-v1.json). All 32 automated
tests passed, including a physical single-object trial. The retention gate
failed; the production controller and trained weights remain unchanged.

### Expanded demonstration experiment

The [predeclared experiment plan](validation/expanded100-plan.json) produced
100 successful demonstrations: a separate copy of the original 30 plus 70 new
successes from 108 attempted training scenes, with the baseline controller.
The completed audit found 369 recorded actions and 301 progress-making actions,
with 100 unique seeds. All failed attempts remain in the collection report.
See the [experiment record](validation/expanded100-results.json) for status and
results; dataset completion alone does not demonstrate a better policy. Data lives
under `robotiq-dataset/mixed-v2-expanded100`, and the new model name is
`mixed-tower-expanded100`. Do not append to or overwrite the historical baseline.
Original demonstrations used preload 0.01; new ones use 0.02. Together with the
different update budget, this makes it a practical expanded-data experiment,
not an isolated causal test of dataset size.

Before training, audit the locally generated dataset and its collection reports:

```shell
python -m ravens.heterogeneous_data_audit --data_dir=robotiq-dataset/mixed-v2-expanded100 --report=robotiq-results/mixed-v2-collection/results.json --report=robotiq-results/expanded100-collection/results.json --output=robotiq-results/my-expanded100-audit.json
python -m unittest ravens.heterogeneous_data_audit_test -v
```

The audit checks matching episode files, unique seeds, array lengths, finite
action poses, unit quaternions, progress labels and successful training-split
provenance. It fingerprints all episode files. Legacy reports must have saved
version-2 manifests when they omit the scene version. Only use trusted local
pickle datasets. The audit does not replace physical evaluation.

Training completed all 3,000 updates with finite losses and saved checkpoints
at 1,000, 2,000 and 3,000 updates. On validation seeds 0-4, these completed
2/5, 3/5 and 2/5 towers respectively. The predeclared selection rule chose
checkpoint 2,000. The matched comparison on validation seeds 5-24 is complete:

| Model | Demonstrations | Complete towers | 95% Wilson interval | Settling errors |
| --- | --- | --- | --- | --- |
| Original `mixed-tower-v2`, checkpoint 1000 | 30 | 15/20 (75%) | 53.1-88.8% | 3 |
| Expanded `mixed-tower-expanded100`, checkpoint 2000 | 100 | 14/20 (70%) | 48.1-85.5% | 4 |

All 20 paired scene manifests match exactly. The models both succeeded on 12
scenes; the expanded model alone succeeded on 2 and the original alone on 3.
These are validation results, not a new held-out test. The expanded run also
had a host/session interruption on seed 13, retained as a timeout failure;
its timing is not comparable. Even a hypothetical success there would only
tie the original at 15/20. This small experiment establishes neither an
improvement nor a statistically reliable regression.

**Decision: retain the original default; do not advance the manipulation
milestone yet.** The expanded model's after-lift grasp rate was higher
(61/71 versus 60/79), but that does not measure retention through transport.
All four of its settling failures ended with the cylinder outside the
workspace, including both short and tall cylinders. Next, isolate grip
retention during lift/transfer across object sizes and trajectories, then
validate any controller change before collecting more demonstrations.
The reserved 100-scene test (test seeds 1000-1099) remains unused.

Reproduction commands (choose a new model name for retraining because the
checkpoints already exist):

```shell
python -m ravens.heterogeneous_train --data_dir=robotiq-dataset/mixed-v2-expanded100 --name=mixed-tower-expanded100 --steps=3000 --save_every=1000 --seed=0 --compile_models
python -m ravens.heterogeneous_run --mode=policy --split=validation --episodes=5 --controller=baseline --name=mixed-tower-expanded100 --checkpoint=1000 --data_dir=robotiq-dataset/mixed-v2-expanded100 --output=robotiq-results/my-expanded100-validation1000
```

To run the selected experimental checkpoint with the visible gripper (using
the existing local weights, not a fresh clone):

```shell
conda activate ravens-transporter
python -m ravens.heterogeneous_run --mode=policy --split=validation --seed=6 --episodes=1 --controller=baseline --name=mixed-tower-expanded100 --checkpoint=2000 --data_dir=robotiq-dataset/mixed-v2-expanded100 --output=robotiq-results/my-expanded100-demo --disp
```

Seed 6 is a previously successful validation example, not an unbiased benchmark.
Use a fresh output directory for every run. A nonzero exit after a completed
evaluation can mean recorded simulation failures; inspect `summary.json` and
`results.json` rather than assuming the model failed to load.

The experiment compared checkpoints 1000, 2000 and 3000 on validation only,
then compared the selected model with the old checkpoint on seeds 5-24.
More data may improve action selection, but cannot directly repair physical grip retention.
Keep failures in the evaluation denominator and do not substitute training loss
for tower-completion results.

### Object set and task progression

| Stage | Objects and initial poses | Required capabilities |
| --- | --- | --- |
| First mixed-object milestone | Cubes, cuboids, and upright cylinders with varying dimensions, random reachable positions, and random yaw | Adaptive grasp width/depth, rotation-aware picks, support-height estimation, stable tower placement |
| Extended benchmark | Hexagonal and triangular prisms, wedges, and sideways objects | Reorientation and selection of usable support faces |
| Generalization benchmark | Held-out object instances, sizes, and selected irregular shapes | Robust perception, grasping, and stability prediction beyond training examples |

Begin with separated objects, then increase clutter and introduce touching or
partially occluded objects. Dimensions must fit the gripper's usable range.
Random yaw does not mean arbitrary 3D orientation; fallen and tilted objects
are a later stage. Meshes must have suitable collision geometry and physical
properties, not only different visual appearances.

The first target is a three-object mixed-shape tower. Keep the existing pyramid
task separate. Not every object collection can form a free-standing tower:
use verified-stackable sets for full-completion tests and a separate challenge
set for the largest stable stack or appropriate stopping.

### Interchangeable policies and shared components

```text
RGB-D observation + task goal
              |
              v
Decision module: Transporter / search / language-guided planner
              |
              v
Candidate grasp and placement -> feasibility checks -> shared executor
                                                        |
                                                        v
                                     Observe, assess stability, and replan
```

Define shared interfaces for observations, goals, candidate actions, transition
prediction, stability evaluation, and execution feedback. Actions should include
pick/place poses and gripper opening. Keep simulator ground truth separate from
camera-derived inputs: it may support demonstrations and outcome scoring, but
must not silently enter vision-only policy inference.

| Approach | Planned role |
| --- | --- |
| Transporter baseline | Predict pick/place actions from RGB-D; extend fixed grasp assumptions for mixed objects and expose multiple action candidates |
| Beam search | Keep top-K partial plans over object order, support choices, and candidate placements |
| Cross-entropy method (CEM) | Refine sampled continuous placement offsets and yaw toward high-scoring feasible candidates |
| Hybrid search | Use Transporter proposals, beam search for discrete choices, and CEM for placement refinement |
| SayCan-style planner | Combine language-based skill usefulness with estimated feasibility; use the shared manipulation controller to execute skills |
| PaLM-E-style / RT-2-style research | Explore multimodal high-level reasoning or a direct vision-language-action policy through appropriate adapters |

These methods operate at different levels, rather than being identical policy
replacements. SayCan-style selection can retain Transporter as a low-level skill.
A direct-action VLA needs compatible robot action encoding and training data;
exact model availability and compute requirements must be checked before
implementation. These are research directions, not bundled model integrations.
References: [Transporter](https://arxiv.org/abs/2010.14406),
[SayCan](https://say-can.github.io/), [PaLM-E](https://palm-e.github.io/), and
[RT-2](https://deepmind.google/blog/rt-2-new-model-translates-vision-and-language-into-action/).

Search requires predicted outcomes, not just stability measured after execution.
Start with isolated simulation rollouts, explicitly labelled as a privileged-state
planning baseline; later evaluate estimated-geometry or learned models. Reject
infeasible actions and score stable task progress, collapse risk, and action cost.
Transporter scores are not calibrated stability probabilities. Execute the first
action of a selected plan, observe the real outcome, and replan.

### End-to-end effectiveness evaluation

The primary experiment starts from a randomized scene and runs without manual
intervention until success, an action/time limit, or an explicit stop. Define a
tower as a connected support structure above one base, with every required object
included for full success. Verify that it remains standing for a fixed settling
period after release. Freeze geometric tolerances, settling duration, and stopping
rules before final testing; use simulator state for scoring, not hidden policy input.

| Metric | Purpose |
| --- | --- |
| Stable tower completion rate | Primary end-to-end success measure |
| Stable objects stacked | Partial completion, including difficult object sets |
| Time to completion and timeout rate | Include perception, planning/model calls, execution, and retries; report failures separately |
| Actions and retries | Execution efficiency |
| Collapse rate and recovery success | Robustness to mistakes and disturbed stacks |
| Grasp and placement success | Diagnostic metrics explaining end-to-end failures |
| Planning time, rollout count, and model calls | Computational cost of each approach |

Use identical held-out scene seeds, objects, robot settings, and stopping rules
across methods. Separate RGB-D-only results from privileged-state planning.
Report known-shape/new-arrangement, unseen-size, unseen-shape, clutter, object-count,
and physical-property variation conditions separately. Hold out test instances
and seeds from demonstration collection and tuning.

Start with a small development suite, then target 100 held-out trials per condition
subject to compute. Report confidence intervals and multiple training seeds where
feasible. Count crashes and timeouts as failures; do not discard unsuccessful runs.
Save videos, action traces, timings, seeds, configuration/checkpoint identifiers,
and failure reasons. Name complete systems explicitly, such as "Transporter + beam
search", to distinguish planner improvements from controller changes.

### Implementation roadmap

1. Add shared policy/action interfaces, a heterogeneous tower task, and the common
   evaluation runner while preserving existing tasks.
2. Validate physical grasps and placements for cubes, cuboids, and cylinders;
   collect varied successful demonstrations and failure data.
3. Train and evaluate a rotation-aware, adaptive-grasp Transporter baseline.
4. Add candidate generation, stability evaluation, and beam-search rollouts.
5. Add CEM placement refinement and compare against the same baseline scenes.
6. Add SayCan-style skill selection; consider direct VLA integration separately.
7. Expand shapes, clutter, reorientation, and closed-loop recovery, repeating the
   held-out end-to-end evaluation at each stage.

## Original Ravens project

Ravens is a collection of simulated tasks in PyBullet for learning vision-based robotic manipulation, with emphasis on pick and place.
It features a Gym-like API with 10 tabletop rearrangement tasks, each with (i) a scripted oracle that provides expert demonstrations (for imitation learning), and (ii) reward functions that provide partial credit (for reinforcement learning).

<img src="docs/tasks.png"><br>

(a) **block-insertion**: pick up the L-shaped red block and place it into the L-shaped fixture.<br>
(b) **place-red-in-green**: pick up the red blocks and place them into the green bowls amidst other objects.<br>
(c) **towers-of-hanoi**: sequentially move disks from one tower to another—only smaller disks can be on top of larger ones.<br>
(d) **align-box-corner**: pick up the randomly sized box and align one of its corners to the L-shaped marker on the tabletop.<br>
(e) **stack-block-pyramid**: sequentially stack 6 blocks into a pyramid of 3-2-1 with rainbow colored ordering.<br>
(f) **palletizing-boxes**: pick up homogeneous fixed-sized boxes and stack them in transposed layers on the pallet.<br>
(g) **assembling-kits**: pick up different objects and arrange them on a board marked with corresponding silhouettes.<br>
(h) **packing-boxes**: pick up randomly sized boxes and place them tightly into a container.<br>
(i) **manipulating-rope**: rearrange a deformable rope such that it connects the two endpoints of a 3-sided square.<br>
(j) **sweeping-piles**: push piles of small objects into a target goal zone marked on the tabletop.<br>

Some tasks require generalizing to unseen objects (d,g,h), or multi-step sequencing with closed-loop feedback (c,e,f,h,i,j).

**Team:** this repository is developed and maintained by [Andy Zeng](https://andyzeng.github.io/), [Pete Florence](http://www.peteflorence.com/), [Daniel Seita](https://people.eecs.berkeley.edu/~seita/), [Jonathan Tompson](https://jonathantompson.github.io/), and [Ayzaan Wahid](https://www.linkedin.com/in/ayzaan-wahid-21676148/). This is the reference repository for the paper:

### Transporter Networks: Rearranging the Visual World for Robotic Manipulation
[Project Website](https://transporternets.github.io/)&nbsp;&nbsp;•&nbsp;&nbsp;[PDF](https://arxiv.org/pdf/2010.14406.pdf)&nbsp;&nbsp;•&nbsp;&nbsp;Conference on Robot Learning (CoRL) 2020

*Andy Zeng, Pete Florence, Jonathan Tompson, Stefan Welker, Jonathan Chien, Maria Attarian, Travis Armstrong,<br>Ivan Krasin, Dan Duong, Vikas Sindhwani, Johnny Lee*

**Abstract.** Robotic manipulation can be formulated as inducing a sequence of spatial displacements: where the space being moved can encompass an object, part of an object, or end effector. In this work, we propose the Transporter Network, a simple model architecture that rearranges deep features to infer spatial displacements from visual input—which can parameterize robot actions. It makes no assumptions of objectness (e.g. canonical poses, models, or keypoints), it exploits spatial symmetries, and is orders of magnitude more sample efficient than our benchmarked alternatives in learning vision-based manipulation tasks: from stacking a pyramid of blocks, to assembling kits with unseen objects; from manipulating deformable ropes, to pushing piles of small objects with closed-loop feedback. Our method can represent complex multi-modal policy distributions and generalizes to multi-step sequential tasks, as well as 6DoF pick-and-place. Experiments on 10 simulated tasks show that it learns faster and generalizes better than a variety of end-to-end baselines, including policies that use ground-truth object poses. We validate our methods with hardware in the real world.

## Installation

**Step 1.** Recommended: install [Miniconda](https://docs.conda.io/en/latest/miniconda.html) with Python 3.7.

```shell
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh -b -u
echo $'\nexport PATH=~/miniconda3/bin:"${PATH}"\n' >> ~/.profile  # Add Conda to PATH.
source ~/.profile
conda init
```

**Step 2.** Create and activate Conda environment, then install GCC and Python packages.

```shell
cd ~/ravens
conda create --name ravens python=3.7 -y
conda activate ravens
sudo apt-get update
sudo apt-get -y install gcc libgl1-mesa-dev
pip install -r requirements.txt
python setup.py install --user
```

**Step 3.** Recommended: install GPU acceleration with NVIDIA [CUDA](https://developer.nvidia.com/cuda-toolkit) 10.1 and [cuDNN](https://developer.nvidia.com/cudnn) 7.6.5 for Tensorflow.
```shell
./oss_scripts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
conda install cudatoolkit==10.1.243 -y
conda install cudnn==7.6.5 -y
```

### Alternative: Pure `pip`

As an example for Ubuntu 18.04:

```shell
./oss_scipts/install_cuda.sh  #  For Ubuntu 16.04 and 18.04.
sudo apt install gcc libgl1-mesa-dev python3.8-venv
python3.8 -m venv ./venv
source ./venv/bin/activate
pip install -U pip
pip install scikit-build
pip install -r ./requirements.txt
export PYTHONPATH=${PWD}
```

## Getting Started

For the Robotiq parallel-gripper stacking task and Transporter agent, see
[Robotiq stacking](docs/robotiq.md). It has its own tested Conda environment
and data-generation commands; the original suction workflow remains below.

**Step 1.** Generate training and testing data (saved locally). Note: remove `--disp` for headless mode.

```shell
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=train --n=10
python ravens/demos.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --mode=test --n=100
```

To run with shared memory, open a separate terminal window and run `python3 -m pybullet_utils.runServer`. Then add `--shared_memory` flag to the command above.

**Step 2.** Train a model e.g., Transporter Networks model. Model checkpoints are saved to the `checkpoints` directory. Optional: you may exit training prematurely after 1000 iterations to skip to the next step.

```shell
python ravens/train.py --task=block-insertion --agent=transporter --n_demos=10
```

**Step 3.** Evaluate a Transporter Networks agent using the model trained for 1000 iterations. Results are saved locally into `.pkl` files.

```shell
python ravens/test.py --assets_root=./ravens/environments/assets/ --disp=True --task=block-insertion --agent=transporter --n_demos=10 --n_steps=1000
```

**Step 4.** Plot and print results.

```shell
python ravens/plot.py --disp=True --task=block-insertion --agent=transporter --n_demos=10
```

**Optional.** Track training and validation losses with Tensorboard.

```shell
python -m tensorboard.main --logdir=logs  # Open the browser to where it tells you to.
```

## Datasets and Pre-Trained Models

Download our generated train and test datasets and pre-trained models.

```shell
wget https://storage.googleapis.com/ravens-assets/checkpoints.zip
wget https://storage.googleapis.com/ravens-assets/block-insertion.zip
wget https://storage.googleapis.com/ravens-assets/place-red-in-green.zip
wget https://storage.googleapis.com/ravens-assets/towers-of-hanoi.zip
wget https://storage.googleapis.com/ravens-assets/align-box-corner.zip
wget https://storage.googleapis.com/ravens-assets/stack-block-pyramid.zip
wget https://storage.googleapis.com/ravens-assets/palletizing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/assembling-kits.zip
wget https://storage.googleapis.com/ravens-assets/packing-boxes.zip
wget https://storage.googleapis.com/ravens-assets/manipulating-rope.zip
wget https://storage.googleapis.com/ravens-assets/sweeping-piles.zip
```

The MDP formulation for each task uses transitions with the following structure:

**Observations:** raw RGB-D images and camera parameters (pose and intrinsics).

**Actions:** a primitive function (to be called by the robot) and parameters.

**Rewards:** total sum of rewards for a successful episode should be =1.

**Info:** 6D poses, sizes, and colors of objects.
