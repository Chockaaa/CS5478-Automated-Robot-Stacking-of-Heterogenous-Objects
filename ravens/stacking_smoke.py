"""Run bounded, seeded oracle trials without TensorFlow. Run as a module."""
import argparse
import copy
import json
import os
import random
import time
from pathlib import Path

import numpy as np
import pybullet as p
from scipy.optimize import linear_sum_assignment
from PIL import Image
from ravens import tasks
from ravens.environments.environment import Environment


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--task', default='stack-block-pyramid-parallel')
  parser.add_argument('--episodes', type=int, default=3)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--assisted', action='store_true')
  parser.add_argument('--snapshots', action='store_true')
  parser.add_argument('--output', default='stacking-results.json')
  args = parser.parse_args()
  if args.episodes < 1:
    parser.error('--episodes must be positive')
  assets = os.path.join(os.path.dirname(__file__), 'environments', 'assets')
  results = []
  env = Environment(assets, disp=args.disp)
  try:
    for seed in range(args.seed, args.seed + args.episodes):
      np.random.seed(seed)
      random.seed(seed)
      env.seed(seed)
      task = tasks.names[args.task]()
      env.set_task(task)
      obs = env.reset()
      env.ee.assisted = args.assisted
      goals = copy.deepcopy(task.goals)
      oracle = task.oracle(env)
      total = 0
      info = env.info
      start = time.monotonic()
      error = None
      for step in range(task.max_steps):
        action = oracle.act(obs, info)
        if action is None:
          break
        try:
          obs, reward, done, info = env.step(action)
        except RuntimeError as exc:
          error = str(exc)
          print(error, flush=True)
          break
        total += reward
        print('seed=%d step=%d reward=%.3f total=%.3f' %
              (seed, step + 1, reward, total), flush=True)
        if hasattr(env.ee, 'contact_history'):
          print('bilateral contacts:', env.ee.contact_history, flush=True)
        if done:
          break
      # Check actual XYZ after two extra seconds, not just Ravens' XY reward.
      for _ in range(480):
        p.stepSimulation()
      errors = []
      supported = True
      upright = True
      for objs, matches, targets, *_ in goals:
        costs = np.full(matches.shape, 1e3)
        for i, (obj, _) in enumerate(objs):
          pos, rot = p.getBasePositionAndOrientation(obj)
          for j, target in enumerate(targets):
            if matches[i, j]:
              costs[i, j] = np.linalg.norm(np.array(pos) - target[0])
          upright &= p.getMatrixFromQuaternion(rot)[8] > .98
          supported &= any(pt[7][2] > .5 and pt[9] > .001
                           for pt in p.getContactPoints(bodyA=obj))
        rows, cols = linear_sum_assignment(costs)
        errors.extend(costs[rows, cols].tolist())
      row = dict(seed=seed, reward=total, steps=step + 1,
                 assisted=args.assisted,
                 error=error,
                 xyz_errors=errors,
                 max_xyz_error=max(errors),
                 upright=bool(upright), supported=bool(supported),
                 success=bool(error is None and total > .99 and max(errors) < .01
                              and upright and supported),
                 seconds=time.monotonic() - start)
      if args.snapshots:
        path = Path(args.output).with_suffix('')
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(env.render()).save(str(path) + '-%d.png' % seed)
      results.append(row)
      print(json.dumps(row), flush=True)
  finally:
    env.close()
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    with open(args.output, 'w') as stream:
      json.dump(dict(task=args.task, results=results), stream, indent=2)
  if not all(row['success'] for row in results):
    raise SystemExit(1)


if __name__ == '__main__':
  main()
