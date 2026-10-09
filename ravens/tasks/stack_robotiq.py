"""Fixed-yaw 40 mm cube pyramid for the original Transporter action space."""
import collections
import numpy as np
import pybullet as p
from ravens.tasks.task import Task
from ravens.tasks.robotiq import Robotiq85, RobotiqPickPlace
from ravens.utils import utils


class StackRobotiq(Task):
  def __init__(self, continuous=False):
    if continuous:
      raise ValueError('Robotiq stacking uses discrete pick/place actions.')
    super().__init__()
    self.ee = Robotiq85
    self.primitive = RobotiqPickPlace()
    self.max_steps = 12
    self.max_settle_steps = 4800

  def reset(self, env):
    super().reset(env)
    p.setPhysicsEngineParameter(numSolverIterations=100)
    base = ((.50, .15, .0025), (0, 0, 0, 1))
    env.add_object('stacking/stand.urdf', base, 'fixed')
    objs = []
    for i, color in enumerate(('purple', 'blue', 'green', 'yellow', 'orange', 'red')):
      pose = ((.35 + .15 * (i % 3) + np.random.uniform(-.01, .01),
               -.15 - .15 * (i // 3) + np.random.uniform(-.01, .01), .02),
              (0, 0, 0, 1))
      obj = env.add_object('stacking/block.urdf', pose)
      p.changeVisualShape(obj, -1, rgbaColor=utils.COLORS[color] + [1])
      objs.append((obj, (np.pi / 2, None)))
    offsets = [(0, -.05, .0225), (0, 0, .0225), (0, .05, .0225),
               (0, -.025, .0625), (0, .025, .0625), (0, 0, .1025)]
    targets = [(utils.apply(base, xyz), base[1]) for xyz in offsets]
    for a, b in ((0, 3), (3, 5), (5, 6)):
      self.goals.append((objs[a:b], np.ones((b-a, b-a)), targets[a:b],
                         False, True, 'pose', None, (b-a)/6))

  def is_match(self, pose0, pose1, symmetry):
    return (super().is_match(pose0, pose1, symmetry) and
            abs(pose0[0][2] - pose1[0][2]) < .008)

  def oracle(self, env, **kwargs):
    del env, kwargs
    def act(obs, info):
      del obs, info
      if not self.goals:
        return None
      objs, matches, targets, *_ = self.goals[0]
      for i, (obj, (sym, _)) in enumerate(objs):
        pose = p.getBasePositionAndOrientation(obj)
        if any(self.is_match(pose, t, sym) for t in targets):
          continue
        for j, target in enumerate(targets):
          if matches[i,j] and not any(self.is_match(
              p.getBasePositionAndOrientation(other), target, sym)
                                     for other, _ in objs):
            # Pick labels describe the top face, place labels the support.
            pick = (tuple(np.array(pose[0]) + (0, 0, .02)), (0, 0, 0, 1))
            yaw = p.getEulerFromQuaternion(pose[1])[2]
            place = (tuple(np.array(target[0]) - (0, 0, .02)),
                     p.getQuaternionFromEuler((0, 0, -yaw)))
            return dict(pose0=pick, pose1=place)
      return None
    return collections.namedtuple('Oracle', ['act'])(act)
