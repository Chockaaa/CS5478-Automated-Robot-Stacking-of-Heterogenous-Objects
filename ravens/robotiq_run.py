"""Bounded Robotiq oracle/data/policy runs. Invoke with python -m."""
import argparse
import copy
import json
import os
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image
from ravens import agents, tasks
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from scipy.optimize import linear_sum_assignment


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=['oracle', 'demos', 'policy'], default='oracle')
  parser.add_argument('--split', choices=['train', 'test'], default='train')
  parser.add_argument('--episodes', dest='n', type=int, default=1)
  parser.add_argument('--attempts', type=int, default=10)
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--data_dir', default='robotiq-dataset')
  parser.add_argument('--train_dir', default='robotiq-training')
  parser.add_argument('--n_demos', type=int, default=10)
  parser.add_argument('--checkpoint', type=int, default=1000)
  parser.add_argument('--max_steps', type=int, default=None)
  parser.add_argument('--output', default='robotiq-results')
  args = parser.parse_args()
  if args.n < 1 or args.attempts < args.n:
    parser.error('Require attempts >= n >= 1')
  assets = os.path.join(os.path.dirname(__file__), 'environments', 'assets')
  task = tasks.names['stack-robotiq']()
  env = Environment(assets, task=task, disp=args.disp)
  policy = task.oracle(env)
  if args.mode == 'policy':
    name = 'stack-robotiq-transporter-robotiq-%d-0' % args.n_demos
    policy = agents.names['transporter-robotiq'](name, 'stack-robotiq', args.train_dir)
    policy.load(args.checkpoint)
  dataset = (Dataset(os.path.join(args.data_dir, 'stack-robotiq-' + args.split))
             if args.mode == 'demos' else None)
  out = Path(args.output)
  out.mkdir(parents=True, exist_ok=True)
  results, saved = [], 0
  try:
    for attempt in range(args.attempts if dataset else args.n):
      seed = (args.seed + attempt) * 2 + int(args.split == 'test')
      if dataset:
        next_seed = (dataset.max_seed // 2 + 1) * 2 + int(args.split == 'test')
        seed = max(seed, next_seed)
      np.random.seed(seed)
      env.seed(seed)
      obs = env.reset()
      goals = copy.deepcopy(task.goals)
      episode, total, reward, error = [], 0., 0., None
      for step in range(args.max_steps or task.max_steps):
        act = policy.act(obs, None)
        if act is None:
          break
        episode.append((obs, act, reward, {}))
        try:
          obs, reward, done, _ = env.step(act)
        except RuntimeError as exc:
          error = str(exc)
          break
        total += reward
        print('seed=%d step=%d reward=%.3f total=%.3f' %
              (seed, step+1, reward, total), flush=True)
        if done:
          break
      episode.append((obs, None, reward, {}))
      for _ in range(480):
        p.stepSimulation()
      errors = []
      for objs, matches, targets, *_ in goals:
        cost = np.array([[np.linalg.norm(np.array(
            p.getBasePositionAndOrientation(obj)[0]) - target[0])
                           for target in targets] for obj, _ in objs])
        ii, jj = linear_sum_assignment(cost)
        errors.extend(cost[ii, jj].tolist())
      success = error is None and total > .99 and max(errors) < .01
      row = dict(seed=seed, reward=total, success=success,
                 max_xyz_error=max(errors), error=error)
      results.append(row)
      print(json.dumps(row), flush=True)
      Image.fromarray(env.render()).save(str(out / ('scene-%d.png' % seed)))
      if dataset and success:
        dataset.add(seed, episode)
        saved += 1
        if saved >= args.n:
          break
  finally:
    env.close()
    (out / 'results.json').write_text(json.dumps(results, indent=2))
  if dataset and saved < args.n:
    raise SystemExit('Only %d/%d successful demos collected.' % (saved, args.n))


if __name__ == '__main__':
  main()
