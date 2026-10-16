"""Development-only oracle motion trace; privileged data is diagnostic only."""
import argparse
import json
from pathlib import Path
import numpy as np
import pybullet as p
from ravens.environments.environment import Environment
from ravens.heterogeneous_run import physics_diagnostics
from ravens.tasks.robotiq import AdaptiveRobotiqPickPlace
from ravens.tasks.stack_heterogeneous import StackHeterogeneous


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--seed', type=int, default=4)
  parser.add_argument('--controller', choices=AdaptiveRobotiqPickPlace.profiles,
                      default='baseline')
  parser.add_argument('--output', required=True)
  args = parser.parse_args()
  out = Path(args.output)
  if out.exists() or args.seed < 0:
    parser.error('Use a new output file and a nonnegative seed')
  out.parent.mkdir(parents=True, exist_ok=True)
  task = StackHeterogeneous(seed=args.seed, split='validation', controller=args.controller)
  env = Environment(str(Path(__file__).parent / 'environments' / 'assets'), task=task)
  trace, error = [], None
  step = 0
  def record(stage, **extra):
    state = physics_diagnostics(task, env)
    state.update(stage=stage, step=step, **extra)
    trace.append(state)
  try:
    np.random.seed(args.seed)
    env.seed(args.seed)
    obs = env.reset()
    record('reset')
    original_move, original_close, original_open = env.movep, env.ee.activate, env.ee.release
    def move(pose, speed, **kwargs):
      result = original_move(pose, speed, **kwargs)
      record('move', target=pose, timeout=result,
             monitored='monitor' in kwargs,
             acceleration=kwargs.get('acceleration'))
      return result
    def close():
      original_close()
      record('close', retained=bool(env.ee.check_grasp()))
    def release():
      record('before_release', retained=bool(env.ee.check_grasp()))
      original_open()
      record('release')
    env.movep, env.ee.activate, env.ee.release = move, close, release
    for step in range(1, 7):
      action = task.oracle(env).act(obs, None)
      if action is None:
        break
      obs, _, _, _ = env.step(action)
      # Match the evaluation runner's post-action settling exactly.
      for _ in range(240):
        p.stepSimulation()
      obs = env._get_obs()
      record('settled')
      if task.evaluate()['success']:
        break
  except Exception as exc:
    error = '%s: %s' % (type(exc).__name__, exc)
    record('failure')
  finally:
    env.close()
    out.write_text(json.dumps(dict(config=vars(args), error=error, trace=trace),
                              indent=2), encoding='utf-8')
  print('Saved %d trace points to %s; error=%s' % (len(trace), out, error))
  if error:
    raise SystemExit(1)


if __name__ == '__main__':
  main()
