"""Single-object diagnostic retention matrix in Ravens, not a learned policy.

Ground-truth poses are used only by this diagnostic oracle and evaluator.
No stacking data or policy weights are modified. Run in its own process.
"""
import argparse
import itertools
import json
import math
from pathlib import Path
import time
import numpy as np
import pybullet as p
from ravens.environments.environment import Environment
from ravens.tasks.task import Task
from ravens.tasks.robotiq import (Robotiq85, SizeAwareRobotiq85,
                                  SizeAwareV2Robotiq85, AdaptiveRobotiqPickPlace)


def cases():
  rows = []
  for position, distance, yaw, scale, family in itertools.product(
      range(2), ('short', 'long'), (0., math.pi/4, math.pi/2),
      range(3), ('cube', 'cuboid', 'cylinder')):
    sizes = {'cube': [(v, v, v) for v in (.032, .041, .050)],
             'cuboid': [(.045, .028, .030), (.0525, .034, .0425), (.060, .040, .055)],
             'cylinder': [(.032, .032, .030), (.041, .041, .0425), (.050, .050, .055)]}
    size = sizes[family][scale]
    rows.append(dict(case_id=len(rows), family=family, size=size,
                     yaw=yaw, position=(.38 if position == 0 else .62, -.23),
                     target=(.5, -.08 if distance == 'short' else .18),
                     distance=distance, scale=scale))
  return rows


def pose_drift(reference, current):
  translation = float(np.linalg.norm(np.array(reference[0])-current[0]))
  dot = abs(float(np.dot(reference[1], current[1])))
  return translation, math.degrees(2 * math.acos(np.clip(dot, 0., 1.)))


class RetentionTask(Task):
  def __init__(self, controller='baseline'):
    super().__init__()
    self.controller = controller
    self.ee = (SizeAwareV2Robotiq85 if controller in ('size-aware-v2',
                                                       'size-aware-v3',
                                                       'size-aware-v4') else
               SizeAwareRobotiq85 if controller == 'size-aware-v1' else Robotiq85)
    self.max_settle_steps = 4800
    self.case = None

  def reset(self, env):
    super().reset(env)
    spec = self.case
    x, y, z = spec['size']
    p.setPhysicsEngineParameter(numSolverIterations=100)
    if spec['family'] == 'cylinder':
      collision = p.createCollisionShape(p.GEOM_CYLINDER, radius=x/2, height=z)
      visual = p.createVisualShape(p.GEOM_CYLINDER, radius=x/2, length=z,
                                   rgbaColor=(.7, .3, .2, 1))
      volume = math.pi * (x/2)**2 * z
    else:
      collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=(x/2, y/2, z/2))
      visual = p.createVisualShape(p.GEOM_BOX, halfExtents=(x/2, y/2, z/2),
                                   rgbaColor=(.3, .6, .8, 1))
      volume = x*y*z
    self.body = p.createMultiBody(baseMass=500*volume,
        baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
        basePosition=(*spec['position'], z/2+.001),
        baseOrientation=p.getQuaternionFromEuler((0, 0, spec['yaw'])))
    env.obj_ids['rigid'].append(self.body)
    p.changeDynamics(self.body, -1, lateralFriction=.8, spinningFriction=.001,
                     restitution=0)
    for _ in range(240):
      p.stepSimulation()


def run_case(env, task, spec):
  task.case = spec
  env.seed(spec['case_id'])
  env.reset()
  row = dict(case=spec, success=False, error=None, first_failure=None, stages=[],
             samples=[], timeout=False)
  stage, reference, ticks = 'approach', None, 0
  original_step = p.stepSimulation

  def sample():
    obj = p.getBasePositionAndOrientation(task.body)
    tool = p.getBasePositionAndOrientation(env.ee.body)
    relative = p.multiplyTransforms(*p.invertTransform(*tool), *obj)
    forces = [sum(pt[9] for pt in p.getContactPoints(
        bodyA=env.ee.body, bodyB=task.body, linkIndexA=pad)) for pad in env.ee.pads]
    drift = pose_drift(reference, relative) if reference else (0., 0.)
    state = dict(stage=stage, tick=ticks, bilateral=all(f > .01 for f in forces),
        normal_forces=forces, relative_pose=relative, translation=drift[0],
        rotation_degrees=drift[1], object_z=obj[0][2],
        driver_angle=p.getJointState(env.ee.body, env.ee.driver)[0],
        pad_bottoms=[p.getAABB(env.ee.body, pad)[0][2] for pad in env.ee.pads])
    row['samples'].append(state)
    if reference is not None and row['first_failure'] is None:
      reason = ('contact_loss' if not state['bilateral'] else
                'translation_slip' if drift[0] > .005 else
                'rotation_slip' if drift[1] > 10 else None)
      if reason:
        row['first_failure'] = dict(stage=stage, tick=ticks, reason=reason)
    return state

  def step(*args, **kwargs):
    nonlocal ticks
    original_step(*args, **kwargs)
    ticks += 1
    if ticks % 12 == 0:
      sample()

  def move(name, xyz):
    nonlocal stage
    stage = name
    if env.movep((xyz, rotation), speed=.002):
      row['timeout'] = True
      raise RuntimeError('Motion timeout at ' + name)
    state = sample()
    row['stages'].append(state)
    return state

  def hold(name, count):
    nonlocal stage
    stage = name
    for _ in range(count):
      p.stepSimulation()
    row['stages'].append(sample())

  started = time.monotonic()
  try:
    p.stepSimulation = step
    xyz, _ = p.getBasePositionAndOrientation(task.body)
    yaw = spec['yaw'] + (math.pi/2 if spec['family'] == 'cuboid' else 0.)
    # Cylinders are rotationally symmetric but grasp approach yaw still varies.
    rotation = p.getQuaternionFromEuler((0, 0, yaw))
    height, width = spec['size'][2], min(spec['size'][:2])
    pick_z = max(.019, xyz[2] + height/2 - .017)
    if task.controller in ('size-aware-v1', 'size-aware-v2', 'size-aware-v3',
                            'size-aware-v4'):
      geometry = {
          'size-aware-v1': AdaptiveRobotiqPickPlace.size_aware_pick_height,
          'size-aware-v2': AdaptiveRobotiqPickPlace.size_aware_v2_pick_height,
          'size-aware-v3': AdaptiveRobotiqPickPlace.size_aware_v3_pick_height,
          'size-aware-v4': AdaptiveRobotiqPickPlace.size_aware_v4_pick_height,
      }[task.controller]
      pick_z = geometry(height, width)
      env.ee.configure_grasp(width)
    row['grasp_geometry'] = dict(height=height, width=width, pick_z=pick_z,
                                 grip_force=env.ee.grip_force)
    move('above_pick', (*xyz[:2], .30))
    move('descend', (*xyz[:2], pick_z))
    hold('preclose', 60)
    stage = 'close'
    env.ee.activate()
    row['stages'].append(sample())
    if not env.ee.check_grasp():
      row['first_failure'] = dict(stage=stage, tick=ticks, reason='no_initial_grasp')
      return row
    state = move('short_lift', (*xyz[:2], pick_z+.04))
    if not state['bilateral'] or state['object_z'] < xyz[2]+.025:
      row['first_failure'] = dict(stage=stage, tick=ticks, reason='failed_short_lift')
      return row
    reference = state['relative_pose']
    hold('short_hold', 120)
    move('high_lift', (*xyz[:2], .30))
    move('transfer', (*spec['target'], .30))
    hold('transport_hold', 480)
    move('lower', (*spec['target'], pick_z+.02))
    row['success'] = row['first_failure'] is None
  except Exception as exc:
    row['error'] = '%s: %s' % (type(exc).__name__, exc)
    if row['first_failure'] is None:
      row['first_failure'] = dict(stage=stage, tick=ticks, reason='exception')
  finally:
    p.stepSimulation = original_step
    row['seconds'] = time.monotonic()-started
    env.ee.release()
  return row


def summarize(rows):
  if not rows:
    raise ValueError('No completed cases')
  families, failures = {}, {}
  for family in ('cube', 'cuboid', 'cylinder'):
    group = [r for r in rows if r['case']['family'] == family]
    successes = sum(r['success'] for r in group)
    families[family] = dict(cases=len(group), successes=successes,
                            rate=successes/len(group) if group else None)
  for row in rows:
    failure = row.get('first_failure')
    if failure:
      key = failure['stage'] + ':' + failure['reason']
      failures[key] = failures.get(key, 0) + 1
  count, successes = len(rows), sum(r['success'] for r in rows)
  rate, z = successes/count, 1.96
  centre = (rate + z*z/(2*count)) / (1+z*z/count)
  half = z * math.sqrt(rate*(1-rate)/count+z*z/(4*count*count)) / (1+z*z/count)
  complete = len({r['case']['case_id'] for r in rows}) == len(cases())
  return dict(cases=count, successes=successes, rate=rate,
              wilson_95=[max(0., centre-half), min(1., centre+half)],
              first_failures=failures,
              families=families, errors=sum(r['error'] is not None for r in rows),
              timeouts=sum(r['timeout'] for r in rows),
              complete_development_matrix=complete,
              development_gate_passed=complete and rate >= .95 and
              all(f['rate'] is not None and f['rate'] >= .90 for f in families.values()))


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument('--limit', type=int, default=108)
  parser.add_argument('--output', required=True)
  parser.add_argument('--controller', choices=('baseline', 'size-aware-v1',
                                                'size-aware-v2', 'size-aware-v3',
                                                'size-aware-v4'),
                      default='baseline')
  parser.add_argument('--disp', action='store_true')
  parser.add_argument('--grip_force', type=float, default=Robotiq85.grip_force)
  parser.add_argument('--preload', type=float, default=Robotiq85.contact_preload)
  parser.add_argument('--pad_friction', type=float, default=Robotiq85.pad_friction)
  parser.add_argument('--pad_spinning', type=float, default=Robotiq85.pad_spinning_friction)
  parser.add_argument('--cases', default='', help='Comma-separated case ids; default uses --limit.')
  args = parser.parse_args()
  out = Path(args.output)
  if out.exists() or not 1 <= args.limit <= 108:
    parser.error('Use a new output directory and a limit from 1 to 108')
  out.mkdir(parents=True)
  task = RetentionTask(args.controller)
  # Preserve controller-specific behavior (for example configure_grasp) while
  # allowing explicit diagnostic physics overrides. Replacing it with the base
  # class would silently disable the controller under test.
  task.ee = type('TunedRobotiq85', (task.ee,), dict(
      grip_force=args.grip_force, contact_preload=args.preload,
      pad_friction=args.pad_friction, pad_spinning_friction=args.pad_spinning))
  env = Environment(str(Path(__file__).parent / 'environments/assets'),
                    task=task, disp=args.disp)
  rows = []
  try:
    selected = ([cases()[int(i)] for i in args.cases.split(',')] if args.cases else cases()[:args.limit])
    for spec in selected:
      try:
        row = run_case(env, task, spec)
      except Exception as exc:
        # Reset or cleanup failures must not disappear from the denominator.
        row = dict(case=spec, success=False, timeout=False,
                   error='%s: %s' % (type(exc).__name__, exc),
                   first_failure=dict(stage='setup_or_cleanup', reason='exception'))
      rows.append(row)
      (out / ('case-%03d.json' % spec['case_id'])).write_text(
          json.dumps(row, indent=2), encoding='utf-8')
      report = dict(config=vars(args), protocol='retention-diagnostic-v1',
                    summary=summarize(rows), episodes=[
                        {k: v for k, v in r.items() if k not in ('samples', 'stages')}
                        for r in rows])
      (out / 'results.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
      print(json.dumps(report['episodes'][-1]), flush=True)
  finally:
    env.close()
  print(json.dumps(summarize(rows)), flush=True)


if __name__ == '__main__':
  main()
