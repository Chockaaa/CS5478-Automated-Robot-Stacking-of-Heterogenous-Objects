# Parallel-gripper stacking

This repo now includes a stacking-only task named
`stack-block-pyramid-parallel`. It uses the bundled Robotiq 2F-85 model while
leaving every original Ravens task, including `stack-block-pyramid`, unchanged.

The original Transporter action still predicts one top-down pick pose and one
place pose. The fingers visibly close, and a fixed constraint stabilizes an
object centered between them. This is intentionally a constraint-assisted
grasp rather than a learned force-closure grasp, so the existing oracle and
Transporter Network can be used without changing their outputs.

## Plan and scope

1. Validate the scripted oracle and gripper in PyBullet.
2. Generate new demonstrations. Old suction demonstrations should not be
   reused because the gripper changes the observations and execution outcome.
3. Train the Transporter agent on the new dataset.
4. Evaluate on separately generated test episodes.
5. After this baseline works, replace the fixed constraint in
   `ParallelJawGripper.activate` with friction-only contact if realistic grasp
   physics is required. That version also needs a grasp-width/orientation
   policy and is outside the current 2D Transporter action space.

## Environment

Ravens pins TensorFlow 2.3 and NumPy 1.18, so use its original Python 3.7
environment on Linux (or WSL2). From the repository root:

```bash
conda create -n ravens-gripper python=3.7 -y
conda activate ravens-gripper
pip install -r requirements.txt
pip install -e .
```

## Run the stacking oracle

Start with one visible episode:

```bash
python ravens/demos.py \
  --assets_root=./ravens/environments/assets/ \
  --disp=True \
  --task=stack-block-pyramid-parallel \
  --mode=train \
  --n=1
```

For headless data generation, remove `--disp=True` and generate both splits:

```bash
python ravens/demos.py --assets_root=./ravens/environments/assets/ \
  --task=stack-block-pyramid-parallel --mode=train --n=100

python ravens/demos.py --assets_root=./ravens/environments/assets/ \
  --task=stack-block-pyramid-parallel --mode=test --n=100
```

## Train and evaluate

```bash
python ravens/train.py \
  --task=stack-block-pyramid-parallel \
  --agent=transporter \
  --n_demos=100

python ravens/test.py \
  --assets_root=./ravens/environments/assets/ \
  --task=stack-block-pyramid-parallel \
  --agent=transporter \
  --n_demos=100 \
  --n_steps=1000
```

Add `--disp=True` to evaluation when you want the PyBullet GUI. Use the same
`--n_demos` and `--n_steps` values used for the checkpoint you want to load.

## Main tuning values

The following fields are in `ravens/tasks/grippers.py`:

- `max_grasp_width = 0.085`: maximum object width in metres.
- `max_center_offset = 0.03`: allowed horizontal error from the jaw centre.
- `_set_fingers(0.72)`: finger closing amount in radians.

The stock blocks are 0.04 m wide, so they are inside these limits.
