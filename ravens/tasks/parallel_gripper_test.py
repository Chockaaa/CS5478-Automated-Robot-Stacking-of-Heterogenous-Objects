"""Contact and release checks independent of the task oracle."""
import os
import unittest
import pybullet as p
from ravens.tasks.parallel_gripper import ParallelJawGripper


class GripperTest(unittest.TestCase):
  def setUp(self):
    p.connect(p.DIRECT)
    robot = p.createMultiBody(baseMass=0)
    self.ids = {'rigid': []}
    assets = os.path.abspath(os.path.join(
        os.path.dirname(__file__), '..', 'environments', 'assets'))
    self.ee = ParallelJawGripper(assets, robot, -1, self.ids)

  def tearDown(self):
    p.disconnect()

  def box(self, x=0):
    shape = p.createCollisionShape(p.GEOM_BOX, halfExtents=(.02, .02, .02))
    obj = p.createMultiBody(.01, shape, basePosition=(x, 0, .115))
    self.ids['rigid'].append(obj)
    return obj

  def test_empty_and_offcentre_do_not_attach(self):
    self.ee.assisted = True
    self.ee.activate()
    self.assertFalse(self.ee.check_grasp())
    self.assertIsNone(self.ee.contact_constraint)
    self.ee.release()
    self.box(x=.15)
    self.ee.activate()
    self.assertFalse(self.ee.check_grasp())
    self.assertIsNone(self.ee.contact_constraint)

  def test_contact_assistance_and_release(self):
    obj = self.box()
    self.ee.assisted = True
    self.ee.activate()
    self.assertTrue(self.ee.check_grasp())
    self.assertEqual(self.ee.grasped_object, obj)
    self.assertIsNotNone(self.ee.contact_constraint)
    self.ee.activate()
    self.assertEqual(p.getNumConstraints(), 2)
    self.ee.release()
    self.assertIsNone(self.ee.contact_constraint)
    self.assertIsNone(self.ee.grasped_object)
    self.assertEqual(p.getNumConstraints(), 1)  # Only the robot mount remains.

  def test_friction_mode_has_no_object_constraint(self):
    self.box()
    self.ee.activate()
    self.assertTrue(self.ee.check_grasp())
    self.assertIsNone(self.ee.contact_constraint)
    self.assertEqual(p.getNumConstraints(), 1)


if __name__ == '__main__':
  unittest.main()
