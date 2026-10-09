# Robotiq gripper and Transporter agent

The two earlier prototype commits were reverted. This replacement uses the
bundled Robotiq 2F-85 link meshes, corrected pad visual scale, collision meshes,
and gear constraints to couple finger joints. Objects are held by contact and
friction; there is no fixed constraint between the gripper and a picked object.
The robot-to-gripper mount and internal joint couplings are constraints.

## Scope and agent

Task: `stack-robotiq`. Agent: `transporter-robotiq`.

This is an initial six-cube pyramid task: cubes are 40 mm and initially upright
with aligned yaw, and pick locations vary. It intentionally matches the original
Transporter's single pick-angle attention model. Arbitrary object sizes,
arbitrary grasp orientations, and irregular objects are not supported yet.

The agent uses the repo's Attention and Transport neural networks:
RGB-D images -> pick pixel -> place pixel/rotation -> Robotiq motion primitive.
Inference does not use oracle object positions or an oracle fallback.
Pick Z is the observed top surface; place Z is the maximum nearby support height
so an empty pixel between pyramid supports does not cause a placement on the
table. The primitive offsets those heights for the known 40 mm cubes and the
Robotiq tool length. Augmentation uses translations, preserving fixed grasp yaw.

The scripted oracle is used only to generate demonstrations and test mechanics.
`--mode=oracle` is NOT a neural policy. `--mode=policy` loads Transporter weights.

## Environment

The `ravens-transporter` Conda environment was created on this computer with
Python 3.10, Conda PyBullet, TensorFlow 2.10.1 and TensorFlow Addons 0.18.0.
TensorFlow currently uses CPU; its CUDA libraries are not installed here.
Existing environments were preserved. To recreate dependencies elsewhere:

```powershell
conda env create -f environment-robotiq.yml
```

Run all commands below from the repository root:

```powershell
conda activate ravens-transporter
python -m unittest ravens.tasks.robotiq_test -v
python -m ravens.robotiq_run --mode=oracle --episodes=1 --disp
```

## Collect demonstrations, train, and evaluate

Generate 10 new successful demonstrations and held-out test demonstrations.
Collection is bounded by `--attempts`; failed episodes are not saved. `episodes`
means new episodes to add, not the total desired dataset size. Seeds are even
for training and odd for testing.

```powershell
python -m ravens.robotiq_run --mode=demos --split=train --episodes=10 --attempts=100 --output=robotiq-results/train
python -m ravens.robotiq_run --mode=demos --split=test --episodes=10 --attempts=100 --output=robotiq-results/test
python -m ravens.train --task=stack-robotiq --agent=transporter-robotiq --data_dir=robotiq-dataset --train_dir=robotiq-training --n_demos=10 --n_steps=1000 --interval=100
python -m ravens.robotiq_run --mode=policy --split=test --episodes=10 --n_demos=10 --checkpoint=1000 --disp --output=robotiq-results/policy
```

These are starting experiment settings, not a promise of success after 1000
updates. Checkpoint names include the demonstration count; use the same count
and training directory for policy loading. Stock suction checkpoints and the
reverted prototype's datasets should not be reused.

## Smoke checks performed

- The physical oracle completed a six-cube pyramid, with final XYZ error below
  4 mm for seed 0. A separate seed 1 run also completed (below 8 mm).
- Additional oracle seeds 2, 4, and 6 all completed, with maximum final errors
  of approximately 4.0, 3.0, and 3.5 mm respectively.
- A successful seed 0 demonstration was saved under `robotiq-dataset`.
- The actual Attention and Transport networks completed a backpropagation step
  on that demonstration and saved `attention-ckpt-1.h5` and
  `transport-ckpt-1.h5` under
  `robotiq-training/checkpoints/stack-robotiq-transporter-robotiq-1-0/`.
- Gripper visual-scale and coupled open/close tests pass.
- Checkpoint reloading, RGB-D inference, and executing a network-selected action
  all completed. That one-action smoke run earned zero reward, as expected for
  a model trained for just one update; it is not a learned-stacking success.

The one-step checkpoint is for integration testing only. It has not learned
stacking. Reproduce its loading/execution smoke check with:

```powershell
python -m ravens.robotiq_run --mode=policy --episodes=1 --n_demos=1 --checkpoint=1 --max_steps=1 --output=robotiq-results/policy-smoke
```

Results and screenshots are saved locally in `robotiq-results/`; datasets,
checkpoints and run artifacts are ignored by Git. This is simulation using a
Robotiq-shaped model and simplified control, not calibrated hardware control.
