"""Regression tests for end-to-end reporting denominators."""
import unittest
from unittest import mock
from types import SimpleNamespace
from ravens.heterogeneous_run import summarize_episodes, physics_diagnostics


class EvaluationTest(unittest.TestCase):
  def test_failure_diagnostics_are_read_only(self):
    task = SimpleNamespace(object_ids=[5],
        manifest={'objects': [{'family': 'cylinder'}]})
    env = SimpleNamespace(ee=SimpleNamespace(body=4, pads=[1, 2]))
    with mock.patch('ravens.heterogeneous_run.p') as physics:
      physics.getBasePositionAndOrientation.return_value = ((.4, .1, .02), (0, 0, 0, 1))
      physics.getBaseVelocity.return_value = ((.3, 0, 0), (0, 12, 0))
      physics.getContactPoints.return_value = []
      physics.getAABB.return_value = ((0, 0, -.002), (0, 0, .02))
      result = physics_diagnostics(task, env)
      self.assertEqual(result['objects'][0]['family'], 'cylinder')
      self.assertEqual(result['objects'][0]['linear_velocity'], (.3, 0, 0))
      self.assertEqual(result['pad_bottoms'], [-.002, -.002])
      physics.stepSimulation.assert_not_called()
      physics.resetBasePositionAndOrientation.assert_not_called()

  def test_errors_are_counted_and_partial_completion_is_separate(self):
    rows = [dict(success=True, stable_objects=3, seconds=10, collapses=0,
                 error=None, actions=[dict(grasp_success=True, stable_objects=3)]),
            dict(success=False, stable_objects=0, seconds=5, collapses=0,
                 error='simulation error', actions=[])]
    summary = summarize_episodes(rows)
    self.assertEqual(summary['episodes'], 2)
    self.assertEqual(summary['success_rate'], .5)
    self.assertEqual(summary['mean_stable_objects'], 1.5)
    self.assertEqual(summary['errors'], 1)
    lo, hi = summary['success_rate_wilson_95']
    self.assertLess(lo, .5)
    self.assertGreater(hi, .5)

  def test_empty_evaluation_is_not_success(self):
    with self.assertRaises(ValueError):
      summarize_episodes([])


if __name__ == '__main__':
  unittest.main()
