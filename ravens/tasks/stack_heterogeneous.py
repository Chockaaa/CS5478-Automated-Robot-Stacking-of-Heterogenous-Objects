"""Dimension-based tower of differently sized boxes and upright cylinders."""
import numpy as np
import pybullet as p
from ravens.tasks.stack_block_pyramid_parallel import StackBlockPyramidParallel
from ravens.tasks.task import Task


class StackHeterogeneous(StackBlockPyramidParallel):
  """Largest footprint first, with known dimensions supplied to the oracle."""
  mixed = False

  def reset(self, env):
    Task.reset(self, env)
    self.max_steps = 9
    top = 0
    xy = (.5 + np.random.uniform(-.04, .04),
          np.random.uniform(-.05, .05))
    for i, width in enumerate((.060, .048, .036)):
      height = np.random.uniform(.032, .045)
      cylinder = self.mixed and i == 1
      shape = p.GEOM_CYLINDER if cylinder else p.GEOM_BOX
      dimensions = (dict(radius=width / 2, height=height) if cylinder else
                    dict(halfExtents=(width / 2, width / 2, height / 2)))
      collision = p.createCollisionShape(shape, **dimensions)
      if cylinder:
        dimensions['length'] = dimensions.pop('height')
      visual = p.createVisualShape(shape, rgbaColor=[i / 3, .5, 1 - i / 3, 1],
                                   **dimensions)
      pose = ((.38 + .11 * i, -.28, height / 2),
              p.getQuaternionFromEuler((0, 0, np.random.uniform(-np.pi, np.pi))))
      obj = p.createMultiBody(.03, collision, visual, *pose)
      env.obj_ids['rigid'].append(obj)
      p.changeDynamics(obj, -1, lateralFriction=1, restitution=0)
      target = ((xy[0], xy[1], top + height / 2), (0, 0, 0, 1))
      self.goals.append(([(obj, (0 if cylinder else np.pi / 2, None))],
                         np.ones((1, 1)), [target], False, True,
                         'pose', None, 1 / 3))
      top += height


class StackMixed(StackHeterogeneous):
  mixed = True
