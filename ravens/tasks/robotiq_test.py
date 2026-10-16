"""Check visible geometry and coupled physical finger motion."""
import os
import unittest
from unittest import mock
import numpy as np
import pybullet as p
from ravens.tasks.robotiq import (Robotiq85, SizeAwareRobotiq85, SizeAwareV2Robotiq85,
                                  AdaptiveRobotiqPickPlace)


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
  def test_size_aware_height_force_and_required_geometry(self):
    self.assertAlmostEqual(
        AdaptiveRobotiqPickPlace.size_aware_pick_height(.03, .032), .024)
    self.assertAlmostEqual(
        AdaptiveRobotiqPickPlace.size_aware_pick_height(.055, .05), .0275)
    primitive = AdaptiveRobotiqPickPlace('size-aware-v1')
    ee, movep = mock.Mock(spec=SizeAwareRobotiq85), mock.Mock(return_value=False)
    ee.grip_force = 3.
    ee.check_grasp.return_value = True
    pose0 = ((.4, -.2, .055), (0, 0, 0, 1))
    pose1 = ((.5, .18, 0), (0, 0, 0, 1))
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      self.assertFalse(primitive(None, movep, ee, pose0, pose1,
                                 grasp_height=.055, grasp_width=.05))
    ee.configure_grasp.assert_called_once_with(.05)
    self.assertAlmostEqual(movep.call_args_list[1].args[0][0][2], .0275)
    missing = AdaptiveRobotiqPickPlace('size-aware-v1')
    movep.reset_mock()
    self.assertFalse(missing(None, movep, ee, pose0, pose1))
    movep.assert_not_called()
    self.assertTrue(missing.last_result['invalid_geometry'])

  def test_size_aware_v2_targets_only_extreme_regimes(self):
    pick = AdaptiveRobotiqPickPlace.size_aware_v2_pick_height
    self.assertAlmostEqual(pick(.03, .032), .022)
    self.assertAlmostEqual(pick(.0425, .041), .0265)
    self.assertAlmostEqual(pick(.055, .04), .039)
    self.assertAlmostEqual(pick(.055, .05), .0275)
    gripper = SizeAwareV2Robotiq85.__new__(SizeAwareV2Robotiq85)
    gripper.configure_grasp(.041)
    self.assertEqual(gripper.grip_force, 2.)
    gripper.configure_grasp(.05)
    self.assertEqual(gripper.grip_force, 3.)

  def test_size_aware_v3_lower_grasp_ablation(self):
    pick = AdaptiveRobotiqPickPlace.size_aware_v3_pick_height
    self.assertAlmostEqual(pick(.03, .032), .019)
    self.assertAlmostEqual(pick(.0425, .041), .0255)
    self.assertAlmostEqual(pick(.055, .04), .038)
    self.assertAlmostEqual(pick(.055, .05), .0275)

  def test_size_aware_v4_matches_spawned_baseline_and_centres_extremes(self):
    pick = AdaptiveRobotiqPickPlace.size_aware_v4_pick_height
    self.assertAlmostEqual(pick(.03, .032), .019)
    self.assertAlmostEqual(pick(.0425, .041), .0265)
    self.assertAlmostEqual(pick(.055, .04), .039)
    self.assertAlmostEqual(pick(.055, .05), .0275)

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

  def test_continuous_monitor_aborts_loaded_motion_before_placement(self):
    primitive = AdaptiveRobotiqPickPlace('monitored-v1')
    ee = mock.Mock()
    ee.check_grasp.return_value = False
    targets = []
    def movep(pose, speed, monitor=None):
      del speed
      targets.append(pose)
      if monitor is not None:
        while monitor():
          pass
      return False
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    self.assertEqual(len(targets), 3)
    self.assertEqual(primitive.last_result['failure_reason'],
                     'lost_grasp_during_lift')
    self.assertEqual(primitive.last_result['monitor_miss_steps'], 12)
    self.assertEqual(ee.release.call_count, 2)

  def test_placement_profile_waypoints_descent_and_holds_before_release(self):
    primitive = AdaptiveRobotiqPickPlace('placement-v1')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.return_value = True
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation') as step:
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    targets = [np.asarray(call.args[0][0]) for call in movep.call_args_list]
    self.assertGreater(len(targets), 20)
    descent = targets[4:-1]
    for start, end in zip(descent, descent[1:]):
      self.assertLessEqual(np.linalg.norm(end-start), .010001)
    self.assertTrue(primitive.last_result['pre_release_hold'])
    self.assertGreaterEqual(step.call_count, 180)

  def test_accel_profile_preserves_geometry_and_limits_loaded_motion(self):
    primitive = AdaptiveRobotiqPickPlace('accel-v1')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.return_value = True
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    calls = movep.call_args_list
    self.assertEqual(len(calls), 6)
    self.assertNotIn('acceleration', calls[0].kwargs)
    self.assertNotIn('acceleration', calls[1].kwargs)
    for call in calls[2:5]:
      self.assertEqual(call.kwargs['acceleration'], .00005)
      self.assertIn('monitor', call.kwargs)
    self.assertNotIn('acceleration', calls[5].kwargs)
    self.assertAlmostEqual(calls[1].args[0][0][2], .023)
    self.assertAlmostEqual(calls[4].args[0][0][2], .105)

  def test_cartesian_accel_profile_limits_only_loaded_tcp_motion(self):
    primitive = AdaptiveRobotiqPickPlace('cartesian-accel-v2')
    ee, movep = mock.Mock(), mock.Mock(return_value=False)
    ee.check_grasp.return_value = True
    with mock.patch('ravens.tasks.robotiq.p.stepSimulation'):
      result = primitive(None, movep, ee,
          ((.4, -.2, .04), (0, 0, 0, 1)),
          ((.5, .18, .08), (0, 0, 0, 1)))
    self.assertFalse(result)
    calls = movep.call_args_list
    self.assertEqual(len(calls), 6)
    for call in calls[:2] + calls[5:]:
      self.assertNotIn('cartesian_speed', call.kwargs)
    for call in calls[2:5]:
      self.assertEqual(call.kwargs['cartesian_speed'], .08)
      self.assertEqual(call.kwargs['cartesian_acceleration'], .4)
      self.assertIn('monitor', call.kwargs)
    self.assertAlmostEqual(calls[1].args[0][0][2], .023)
    self.assertAlmostEqual(calls[4].args[0][0][2], .105)

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
