"""Check visible geometry and coupled physical finger motion."""
import os
import unittest
import pybullet as p
from ravens.tasks.robotiq import Robotiq85


class RobotiqTest(unittest.TestCase):
  def setUp(self):
    p.connect(p.DIRECT)
    robot = p.createMultiBody(baseMass=0)
    assets = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'environments', 'assets'))
    self.ee = Robotiq85(assets, robot, -1, {'rigid': []})

  def tearDown(self):
    p.disconnect()

  def test_visible_meshes_and_pad_scale(self):
    visuals = p.getVisualShapeData(self.ee.body)
    self.assertGreaterEqual(len(visuals), 10)
    pads = [v for v in visuals if v[1] in self.ee.pads]
    self.assertEqual(len(pads), 2)
    for pad in pads:
      self.assertEqual(pad[3], (.001, .001, .001))

  def test_empty_close_and_reopen(self):
    before = p.getJointState(self.ee.body, self.ee.driver)[0]
    count = p.getNumConstraints()
    self.ee.activate()
    closed = p.getJointState(self.ee.body, self.ee.driver)[0]
    self.assertGreater(closed, before + .2)
    self.assertFalse(self.ee.check_grasp())
    self.assertEqual(p.getNumConstraints(), count)
    self.ee.release()
    self.assertLess(p.getJointState(self.ee.body, self.ee.driver)[0], closed - .2)


if __name__ == '__main__':
  unittest.main()
