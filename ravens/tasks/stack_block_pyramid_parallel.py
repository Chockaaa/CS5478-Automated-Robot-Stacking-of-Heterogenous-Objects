# coding=utf-8
"""Stack-block-pyramid variant using a parallel-jaw gripper."""

import collections
import numpy as np
import pybullet as p
from ravens.tasks.parallel_gripper import ParallelJawGripper, ParallelPickPlace
from ravens.tasks.stack_block_pyramid import StackBlockPyramid
from ravens.tasks.task import Task
from ravens.utils import utils


class StackBlockPyramidParallel(StackBlockPyramid):
  """Six-block pyramid with motor-driven parallel fingers."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.continuous:
      raise ValueError(
          'The parallel-gripper task supports discrete actions only.')
    self.ee = ParallelJawGripper
    self.primitive = ParallelPickPlace()

  def reset(self, env):
    Task.reset(self, env)
    # Reserve jaw clearance between scattered objects and the assembly area.
    base = ((.5, .15, .0025), p.getQuaternionFromEuler(
        (0, 0, np.random.uniform(-np.pi, np.pi))))
    env.add_object('stacking/stand.urdf', base, 'fixed')
    objs = []
    colors = ['purple', 'blue', 'green', 'yellow', 'orange', 'red']
    for i in range(6):
      pos = (.35 + .15 * (i % 3) + np.random.uniform(-.01, .01),
             -.15 - .15 * (i // 3) + np.random.uniform(-.01, .01), .02)
      rot = p.getQuaternionFromEuler((0, 0, np.random.uniform(-np.pi, np.pi)))
      obj = env.add_object('stacking/block.urdf', (pos, rot))
      p.changeVisualShape(obj, -1, rgbaColor=utils.COLORS[colors[i]] + [1])
      objs.append((obj, (np.pi / 2, None)))
    locations = [(0, -.05, .0225), (0, 0, .0225), (0, .05, .0225),
                 (0, -.025, .0625), (0, .025, .0625), (0, 0, .1025)]
    targets = [(utils.apply(base, pos), base[1]) for pos in locations]
    for start, end in ((0, 3), (3, 5), (5, 6)):
      count = end - start
      self.goals.append((objs[start:end], np.ones((count, count)),
                         targets[start:end], False, True, 'pose', None,
                         count / 6))

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
        if any(self.is_match(pose, target, sym) for target in targets):
          continue
        for j, target in enumerate(targets):
          if matches[i, j] and not any(
              self.is_match(p.getBasePositionAndOrientation(other), target, sym)
              for other, _ in objs):
            yaw = p.getEulerFromQuaternion(pose[1])[2]
            return dict(pose0=(pose[0], p.getQuaternionFromEuler((0, 0, yaw))),
                        pose1=target)
      return None
    return collections.namedtuple('Oracle', ['act'])(act)
