"""Mixed-object tower scenes, physical oracle, and support-based scoring."""
import collections
import numpy as np
import pybullet as p
from ravens.tasks.task import Task
from ravens.tasks.robotiq import (Robotiq85, RetentionRobotiq85,
                                  SizeAwareRobotiq85, SizeAwareV2Robotiq85,
                                  AdaptiveRobotiqPickPlace)


SPLITS = {'train': 0, 'validation': 1, 'test': 2}
SCENE_VERSION = 2


def sample_scene(seed=0, split='train'):
  """Return a JSON-compatible manifest, independent of global RNG state.

  Splits use distinct random streams, not held-out shape families. Dimensions
  are full XYZ extents in metres (cylinder X/Y are its diameter). These scenes
  are candidates for stacking, not yet physically verified-stackable episodes.
  """
  if split not in SPLITS:
    raise ValueError('Unknown split: %s' % split)
  if not isinstance(seed, (int, np.integer)) or seed < 0:
    raise ValueError('seed must be a non-negative integer')
  rng = np.random.default_rng(np.random.SeedSequence([int(seed), SPLITS[split]]))
  objects = []
  for family in rng.permutation(['cube', 'cuboid', 'cylinder']):
    if family == 'cube':
      side = float(rng.uniform(.032, .050))
      size = [side] * 3
    elif family == 'cuboid':
      size = [float(rng.uniform(.045, .060)),
              float(rng.uniform(.028, .040)), float(rng.uniform(.030, .055))]
    else:
      diameter = float(rng.uniform(.032, .050))
      size = [diameter, diameter, float(rng.uniform(.030, .055))]
    radius = size[0] / 2 if family == 'cylinder' else np.linalg.norm(size[:2]) / 2
    for _ in range(1000):
      xy = rng.uniform([.32, -.34], [.68, -.08])
      if all(np.linalg.norm(xy - np.array(o['position'][:2])) >
             radius + o['footprint_radius'] + .035 for o in objects):
        break
    else:
      raise RuntimeError('Could not sample separated objects')
    volume = (np.pi * (size[0]/2)**2 * size[2] if family == 'cylinder'
              else np.prod(size))
    objects.append(dict(
        name='object-%d' % len(objects), family=str(family), size=size,
        position=[float(xy[0]), float(xy[1]), size[2]/2 + .001],
        yaw=float(rng.uniform(-np.pi, np.pi)), footprint_radius=float(radius),
        mass=float(500 * volume), friction=.8,
        color=rng.uniform(.2, .9, size=3).tolist() + [1.]))
  return dict(schema_version=SCENE_VERSION, seed=int(seed), split=split, objects=objects,
              goal=dict(type='tower', base_xy=[.5, .18], object_count=3),
              target_marker=dict(size=[.08, .08], collision=False),
              status='upright mixed-object tower baseline')


class StackHeterogeneous(Task):
  """Three-object tower, with ground truth restricted to oracle/evaluation."""
  def __init__(self, continuous=False, seed=0, split='train', controller='baseline'):
    if continuous:
      raise ValueError('Mixed-object towers require discrete actions.')
    super().__init__()
    self.ee = (RetentionRobotiq85 if controller == 'retention-v1' else
               SizeAwareV2Robotiq85 if controller in ('size-aware-v2',
                                                       'size-aware-v3',
                                                       'size-aware-v4') else
               SizeAwareRobotiq85 if controller == 'size-aware-v1' else Robotiq85)
    self.primitive = AdaptiveRobotiqPickPlace(controller)
    self.scene_seed = seed
    self.mode = split
    self.max_steps = 12
    self.max_settle_steps = 4800

  def reset(self, env):
    super().reset(env)
    self.manifest = sample_scene(self.scene_seed, self.mode)
    self.object_ids = []
    p.setPhysicsEngineParameter(numSolverIterations=100)
    # Camera-visible goal cue, not a support fixture or a hidden policy input.
    marker = p.createVisualShape(p.GEOM_BOX, halfExtents=[.04, .04, .0001],
                                 rgbaColor=[.15, .65, .70, 1.])
    self.target_marker = p.createMultiBody(
        baseMass=0, baseCollisionShapeIndex=-1, baseVisualShapeIndex=marker,
        basePosition=[*self.manifest['goal']['base_xy'], .0001])
    for obj in self.manifest['objects']:
      if obj['family'] == 'cylinder':
        collision = p.createCollisionShape(p.GEOM_CYLINDER,
            radius=obj['size'][0]/2, height=obj['size'][2])
        visual = p.createVisualShape(p.GEOM_CYLINDER,
            radius=obj['size'][0]/2, length=obj['size'][2], rgbaColor=obj['color'])
      else:
        half = np.array(obj['size']) / 2
        collision = p.createCollisionShape(p.GEOM_BOX, halfExtents=half)
        visual = p.createVisualShape(p.GEOM_BOX, halfExtents=half,
                                     rgbaColor=obj['color'])
      body = p.createMultiBody(baseMass=obj['mass'],
          baseCollisionShapeIndex=collision, baseVisualShapeIndex=visual,
          basePosition=obj['position'],
          baseOrientation=p.getQuaternionFromEuler((0, 0, obj['yaw'])))
      p.changeDynamics(body, -1, lateralFriction=obj['friction'],
                       spinningFriction=.001, restitution=0)
      env.obj_ids['rigid'].append(body)
      self.object_ids.append(body)
    # Newly spawned bodies have zero velocity even while above the plane.
    # Settle before the first camera observation and oracle height label.
    for _ in range(240):
      p.stepSimulation()

  def oracle(self, env, **kwargs):
    del env, kwargs
    def act(obs, info):
      del obs, info
      evaluation = self.evaluate()
      stacked = evaluation['stacked_ids']
      candidates = [i for i, body in enumerate(self.object_ids) if body not in stacked]
      if not candidates:
        return None
      # Broad supports first; all placements are centred, preserve object yaw.
      index = max(candidates, key=lambda i: (
          self.manifest['objects'][i]['family'] != 'cylinder',
          np.prod(self.manifest['objects'][i]['size'][:2])))
      obj = self.manifest['objects'][index]
      xyz, rotation = p.getBasePositionAndOrientation(self.object_ids[index])
      yaw = p.getEulerFromQuaternion(rotation)[2] if obj['family'] != 'cylinder' else 0.
      if obj['family'] == 'cuboid':
        yaw += np.pi / 2  # Tool convention: jaws close along commanded world X.
      period = np.pi / 2 if obj['family'] == 'cube' else np.pi
      yaw = (yaw + period/2) % period - period/2
      rotation = p.getQuaternionFromEuler((0, 0, yaw))
      pick = ((xyz[0], xyz[1], xyz[2] + obj['size'][2]/2), rotation)
      support = evaluation['top_height']
      base_xy = self.manifest['goal']['base_xy']
      place = ((*base_xy, support), rotation)
      return dict(pose0=pick, pose1=place, grasp_height=obj['size'][2],
                  grasp_width=min(obj['size'][:2]))
    return collections.namedtuple('Oracle', ['act'])(act)

  def evaluate(self):
    """Score a centred upright support chain; no fixed object-order targets.

    Caller performs the final temporal hold test. Position, contact, tilt and
    velocity checks here use privileged simulator state ONLY for evaluation.
    """
    records = []
    for body, obj in zip(self.object_ids, self.manifest['objects']):
      xyz, quat = p.getBasePositionAndOrientation(body)
      linear, angular = p.getBaseVelocity(body)
      rotation = np.array(p.getMatrixFromQuaternion(quat)).reshape(3, 3)
      vertical = (np.max(np.abs(rotation[2])) if obj['family'] == 'cube'
                  else abs(rotation[2, 2]))
      upright = vertical > np.cos(np.deg2rad(10))
      records.append((xyz[2], body, obj, np.array(xyz), upright,
                      np.linalg.norm(linear) < .01 and np.linalg.norm(angular) < .1))
    chain, height = [], 0.
    for _, body, obj, xyz, upright, static in sorted(records):
      base_xy = self.manifest['goal']['base_xy']
      if not upright or not static or np.linalg.norm(xyz[:2] - base_xy) > .012:
        continue
      bottom = xyz[2] - obj['size'][2]/2
      if abs(bottom - height) > .004:
        continue
      if chain and not p.getContactPoints(bodyA=body, bodyB=chain[-1]):
        continue
      chain.append(body)
      height = xyz[2] + obj['size'][2]/2
    return dict(stacked_ids=chain, stable_objects=len(chain),
                top_height=float(height), success=len(chain) == len(self.object_ids))

  def reward(self):
    value = self.evaluate()['stable_objects'] / len(self.object_ids)
    delta = value - self._rewards
    self._rewards = value
    return delta, {}

  def done(self):
    return self.evaluate()['success']
