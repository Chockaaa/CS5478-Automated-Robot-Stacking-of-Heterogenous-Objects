# coding=utf-8
"""Stack-block-pyramid variant using a parallel-jaw gripper."""

from ravens.tasks.grippers import ParallelJawGripper
from ravens.tasks.stack_block_pyramid import StackBlockPyramid


class StackBlockPyramidParallel(StackBlockPyramid):
  """Original six-block stacking task with a Robotiq 2F-85 gripper."""

  def __init__(self, *args, **kwargs):
    super().__init__(*args, **kwargs)
    if self.continuous:
      raise ValueError(
          'The parallel-gripper task supports discrete actions only.')
    self.ee = ParallelJawGripper
