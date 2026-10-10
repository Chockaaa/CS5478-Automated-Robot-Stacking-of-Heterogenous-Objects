"""Collect/evaluate mixed towers; never pass privileged info to learned agents."""
import argparse
import hashlib
import json
import time
from pathlib import Path
import numpy as np
import pybullet as p
from PIL import Image
from ravens.dataset import Dataset
from ravens.environments.environment import Environment
from ravens.tasks.stack_heterogeneous import StackHeterogeneous, SPLITS, SCENE_VERSION
from ravens.tasks.robotiq import AdaptiveRobotiqPickPlace


def physics_diagnostics(task, env):
  """Privileged failure evidence; never supplied to the policy or controller."""
  objects = []
  for body, spec in zip(task.object_ids, task.manifest['objects']):
    pose = p.getBasePositionAndOrientation(body)
    linear, angular = p.getBaseVelocity(body)
    objects.append(dict(body=body, family=spec['family'], pose=pose,
                        linear_velocity=linear, angular_velocity=angular,
                        contacts=[dict(body=pt[2], force=pt[9], distance=pt[8])
                                  for pt in p.getContactPoints(bodyA=body)]))
  return dict(objects=objects, pad_bottoms=[
      p.getAABB(env.ee.body, pad)[0][2] for pad in env.ee.pads])


def summarize_episodes(rows, saved=0):
  """Include every requested episode, including exceptions, in the denominator."""
  if not rows:
    raise ValueError('Cannot summarize an empty evaluation')
  count = len(rows)
  successes = sum(r['success'] for r in rows)
  fraction = successes / count
  z = 1.96
  centre = (fraction + z*z/(2*count)) / (1+z*z/count)
  half = z * np.sqrt(fraction*(1-fraction)/count + z*z/(4*count*count)) / (1+z*z/count)
  actions = [a for row in rows for a in row['actions']]
  grasps = sum(a.get('grasp_success', False) for a in actions)
  improvements = 0
  for row in rows:
    previous = 0
    for action in row['actions']:
      current = action.get('stable_objects', previous)
      improvements += int(current > previous)
      previous = current
  return dict(
      episodes=count, successes=successes, success_rate=fraction,
      success_rate_wilson_95=[max(0., float(centre-half)), min(1., float(centre+half))],
      mean_stable_objects=float(np.mean([r['stable_objects'] for r in rows])),
      mean_episode_seconds=float(np.mean([r['seconds'] for r in rows])),
      mean_actions=len(actions)/count, actions=len(actions), grasp_successes=grasps,
      grasp_success_rate=grasps/len(actions) if actions else 0.,
      actions_increasing_stable_count=improvements,
      collapses=sum(r['collapses'] for r in rows),
      time_limits=sum(r.get('time_limit_reached', False) for r in rows),
      motion_timeouts=sum(a.get('timeout', False) for a in actions),
      saved_demos=saved, errors=sum(r['error'] is not None for r in rows))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--mode', choices=['oracle', 'demos', 'policy'], default='oracle')
  parser.add_argument('--split', choices=list(SPLITS), default='train')
  parser.add_argument('--seed', type=int, default=0)
  parser.add_argument('--episodes', type=int, default=5)
  parser.add_argument('--max_steps', type=int, default=6)
  parser.add_argument('--max_seconds', type=float, default=180.)
  parser.add_argument('--attempts', type=int, default=100)
  parser.add_argument('--output', default='robotiq-results/heterogeneous-run')
  parser.add_argument('--data_dir', default='robotiq-dataset/mixed-v2')
  parser.add_argument('--train_dir', default='robotiq-training')
  parser.add_argument('--checkpoint', type=int, default=1000)
  parser.add_argument('--name', default='mixed-tower-v2')
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--controller', choices=AdaptiveRobotiqPickPlace.profiles,
                      default='baseline')
  args = parser.parse_args()
  if args.seed < 0 or min(args.episodes, args.max_steps, args.attempts) < 1 or args.max_seconds <= 0:
    parser.error('Require nonnegative seed and positive counts')
  out = Path(args.output)
  if (out / 'results.json').exists():
    parser.error('Results already exist: choose a new --output directory')
  out.mkdir(parents=True, exist_ok=True)
  dataset = Dataset(str(Path(args.data_dir) / ('stack-heterogeneous-' + args.split)))
  if args.mode == 'demos':
    args.seed = max(args.seed, dataset.max_seed + 1)
  policy = None
  if args.mode == 'policy':
    import tensorflow as tf
    tf.config.threading.set_intra_op_parallelism_threads(4)
    tf.config.threading.set_inter_op_parallelism_threads(2)
    for gpu in tf.config.list_physical_devices('GPU'):
      tf.config.experimental.set_memory_growth(gpu, True)
    from ravens.agents.transporter_heterogeneous import HeterogeneousTransporterAgent
    policy = HeterogeneousTransporterAgent(args.name, 'stack-heterogeneous', args.train_dir)
    policy.load(args.checkpoint)
  args.decoder = policy.decoder_version if policy is not None else 'ground-truth oracle'
  args.scene_version = SCENE_VERSION
  args.controller_version = ('robotiq-bilateral-contact-hold-v3' if
                             args.controller == 'baseline' else 'robotiq-' + args.controller)
  if policy is not None:
    checkpoint_dir = Path(args.train_dir) / 'checkpoints' / args.name
    args.checkpoint_sha256 = {
        name: hashlib.sha256((checkpoint_dir / (
            '%s-ckpt-%d.h5' % (name, args.checkpoint))).read_bytes()).hexdigest()
        for name in ('attention', 'transport')}
  task = StackHeterogeneous(split=args.split, controller=args.controller)
  args.contact_preload = task.ee.contact_preload
  env = Environment(str(Path(__file__).parent / 'environments' / 'assets'),
                    task=task, disp=args.disp)
  rows, saved = [], 0
  try:
    for seed in range(args.seed, args.seed + (args.attempts if args.mode == 'demos' else args.episodes)):
      started = time.perf_counter()
      row = dict(seed=seed, split=args.split, mode=args.mode, success=False,
                 actions=[], error=None, collapses=0, time_limit_reached=False)
      episode, reward, prev_count = [], 0., 0
      try:
        task.scene_seed = seed
        np.random.seed(seed)
        env.seed(seed)
        obs = env.reset()
        (out / ('manifest-%d.json' % seed)).write_text(
            json.dumps(task.manifest, indent=2), encoding='utf-8')
        actor = policy if policy is not None else task.oracle(env)
        for step in range(args.max_steps):
          if time.perf_counter() - started >= args.max_seconds:
            row['time_limit_reached'] = True
            break
          before = time.perf_counter()
          action = actor.act(obs, None)
          infer_time = time.perf_counter() - before
          if action is None:
            break
          episode.append((obs, action, reward, {}))
          details = dict(step=step+1, grasp_success=False,
                         inference_seconds=infer_time,
                         pose0=[[float(v) for v in part] for part in action['pose0']],
                         pose1=[[float(v) for v in part] for part in action['pose1']])
          row['actions'].append(details)
          obs, reward, _, _ = env.step(action)
          # Fixed post-action settling includes angular motion.
          for _ in range(240):
            p.stepSimulation()
          obs = env._get_obs()  # Next decision must see the post-settling state.
          metrics = task.evaluate()
          made_progress = metrics['stable_objects'] > prev_count
          if metrics['stable_objects'] < prev_count:
            row['collapses'] += 1
          prev_count = metrics['stable_objects']
          details.update(task.primitive.last_result)
          details['stable_objects'] = prev_count
          old_obs, old_action, old_reward, _ = episode[-1]
          episode[-1] = (old_obs, old_action, old_reward,
                         dict(made_progress=made_progress))
          # Privileged diagnostics are logged AFTER execution, never policy inputs.
          details['object_poses'] = [p.getBasePositionAndOrientation(body)
                                    for body in task.object_ids]
          details['physics'] = physics_diagnostics(task, env)
          print('seed=%d step=%d stacked=%d grasp=%s' %
                (seed, step+1, prev_count, details.get('grasp_success')), flush=True)
          if metrics['success'] or details.get('timeout'):
            break
        # Two-second hold, sampled every 0.1 seconds; all checks must pass.
        counts = [task.evaluate()['stable_objects']]
        for tick in range(480):
          p.stepSimulation()
          if tick % 24 == 23:
            counts.append(task.evaluate()['stable_objects'])
        row['stable_objects'] = min(counts)
        row['time_limit_reached'] |= time.perf_counter() - started >= args.max_seconds
        row['success'] = (min(counts) == 3 and not row['time_limit_reached'] and
                          not any(a.get('timeout') for a in row['actions']))
        episode.append((obs, None, reward, {}))
        if args.mode == 'demos' and row['success']:
          dataset.add(seed, episode)
          saved += 1
        Image.fromarray(env.render()).save(str(out / ('scene-%d.png' % seed)))
      except Exception as exc:  # Failed episodes remain in the denominator.
        row['success'] = False
        if row['actions']:
          row['actions'][-1].update(task.primitive.last_result)
        row['error'] = '%s: %s' % (type(exc).__name__, exc)
        row['stable_objects'] = 0
        # Preserve the actual failing state, including unsettled/fallen bodies.
        # Diagnostic capture must not hide the original exception.
        try:
          row['failure_physics'] = physics_diagnostics(task, env)
          Image.fromarray(env.render()).save(str(out / ('failure-%d.png' % seed)))
        except Exception as diagnostic_exc:
          row['diagnostic_error'] = str(diagnostic_exc)
      row['seconds'] = time.perf_counter() - started
      rows.append(row)
      (out / 'results.json').write_text(json.dumps(dict(config=vars(args), episodes=rows), indent=2), encoding='utf-8')
      print(json.dumps({k: v for k, v in row.items()
                        if k not in ('actions', 'failure_physics')}), flush=True)
      if args.mode == 'demos' and saved >= args.episodes:
        break
  finally:
    env.close()
  summary = summarize_episodes(rows, saved)
  (out / 'summary.json').write_text(json.dumps(summary, indent=2), encoding='utf-8')
  print(json.dumps(summary), flush=True)
  if summary['errors'] or (args.mode == 'demos' and saved < args.episodes):
    raise SystemExit(1)


if __name__ == '__main__':
  main()
