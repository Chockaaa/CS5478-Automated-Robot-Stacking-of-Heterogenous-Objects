"""Check visible geometry and coupled physical finger motion."""
import os
import unittest
from unittest import mock
import numpy as np
import pybullet as p
from ravens.tasks.robotiq import Robotiq85, AdaptiveRobotiqPickPlace


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
    self.assertGreater(closed, .75)
    self.assertFalse(self.ee.check_grasp())
    self.assertEqual(p.getNumConstraints(), count)
    self.ee.release()
    self.assertLess(p.getJointState(self.ee.body, self.ee.driver)[0], closed - .2)
    self.assertLess(abs(p.getJointState(self.ee.body, self.ee.driver)[0]), .02)


class AdaptiveMotionTest(unittest.TestCase):
  def test_cartesian_loaded_waypoints_and_retention_logging(self):
    primitive = AdaptiveRobotiqPickPlace('cartesian-v1')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.return_value = True
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    targets = [np.asarray(call.args[0][0]) for call in movep.call_args_list]
    self.assertGreater(len(targets), 50)
    for start, end in zip(targets[1:-2], targets[2:-1]):
      self.assertLessEqual(np.linalg.norm(end-start), .010001)
    np.testing.assert_allclose(targets[-2], [.5, .18, .105])
    self.assertTrue(primitive.last_result['after_transfer'])
    self.assertTrue(primitive.last_result['before_release'])

  def test_cartesian_timeout_stops_waypoints(self):
    primitive = AdaptiveRobotiqPickPlace('cartesian-v1')
    ee, movep = mock.Mock(), mock.Mock(side_effect=[False, False, True])
    pose = ((.4, -.2, .04), (0, 0, 0, 1))
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      self.assertTrue(primitive(None, movep, ee, pose, pose))
    self.assertEqual(movep.call_count, 3)
    self.assertTrue(primitive.last_result['timeout'])
    self.assertEqual(ee.release.call_count, 2)

  def test_cartesian_rotation_change_rejected_before_motion(self):
    primitive = AdaptiveRobotiqPickPlace('cartesian-v1')
    ee, movep = mock.Mock(), mock.Mock()
    self.assertFalse(primitive(None, movep, ee,
        ((.4, -.2, .04), (0, 0, 0, 1)),
        ((.5, .18, 0), (0, 0, 1, 0))))
    movep.assert_not_called()
    self.assertFalse(primitive.last_result['timeout'])
    self.assertEqual(primitive.last_result['failure_reason'],
                     'unsupported_loaded_rotation')

  def test_unknown_profile_rejected(self):
    with self.assertRaises(ValueError):
      AdaptiveRobotiqPickPlace('unknown')

  def test_clearance_profile_keeps_short_grasp_higher(self):
    primitive = AdaptiveRobotiqPickPlace('clearance-v1')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.return_value = True
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      primitive(None, movep, ee, ((.4, -.2, .03), (0, 0, 0, 1)),
                ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertAlmostEqual(movep.call_args_list[1].args[0][0][2], .027)
    self.assertAlmostEqual(movep.call_args_list[4].args[0][0][2], .109)

  def test_guarded_transfer_loss_skips_placement_descent(self):
    primitive = AdaptiveRobotiqPickPlace('guarded-v1')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.side_effect = [True, True, True, False]
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    self.assertEqual(movep.call_count, 5)
    self.assertTrue(primitive.last_result['grasp_success'])
    self.assertFalse(primitive.last_result['after_transfer'])
    self.assertEqual(primitive.last_result['failure_reason'],
                     'lost_grasp_after_transfer')
    self.assertEqual(ee.release.call_count, 2)

  def test_contact_grasp_and_retreat(self):
    primitive = AdaptiveRobotiqPickPlace()
    ee = mock.Mock()
    ee.check_grasp.return_value = True
    targets = []
    def movep(pose, speed):
      targets.append(np.asarray(pose[0]))
      return False
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      timeout = primitive(None, movep, ee,
          ((.35, -.25, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(timeout)
    self.assertTrue(primitive.last_result['grasp_success'])
    self.assertEqual(len(targets), 6)
    np.testing.assert_allclose(targets[-1], [.5, .18, .3])
    ee.activate.assert_called_once()
    self.assertEqual(ee.release.call_count, 2)

  def test_timeout_releases_and_invalid_height_never_moves(self):
    primitive = AdaptiveRobotiqPickPlace()
    ee, movep = mock.Mock(), mock.Mock(return_value=True)
    pose = ((.4, -.2, .04), (0, 0, 0, 1))
    self.assertTrue(primitive(None, movep, ee, pose, pose))
    self.assertTrue(primitive.last_result['timeout'])
    self.assertEqual(ee.release.call_count, 2)
    movep.reset_mock()
    invalid = ((.4, -.2, .2), (0, 0, 0, 1))
    self.assertFalse(primitive(None, movep, ee, invalid, pose))
    movep.assert_not_called()
    self.assertTrue(primitive.last_result['invalid_height'])


if __name__ == '__main__':
  unittest.main()
