"""Retention protocol invariants, independent of policy training."""
import unittest
from pathlib import Path
import pybullet as p
from ravens.environments.environment import Environment
from ravens.retention_benchmark import cases, pose_drift, summarize, RetentionTask, run_case


class RetentionBenchmarkTest(unittest.TestCase):
  def test_balanced_matrix(self):
    rows = cases()
    self.assertEqual(len(rows), 108)
    self.assertEqual([r['case_id'] for r in rows], list(range(108)))
    for family in ('cube', 'cuboid', 'cylinder'):
      group = [r for r in rows if r['family'] == family]
      self.assertEqual(len(group), 36)
      self.assertEqual(len({tuple(r['size']) for r in group}), 3)
      self.assertEqual(len({r['yaw'] for r in group}), 3)
      self.assertEqual(len({r['position'] for r in group}), 2)
      self.assertEqual(len({r['target'] for r in group}), 2)

  def test_quaternion_sign_and_translation(self):
    distance, angle = pose_drift(((0, 0, 0), (0, 0, 0, 1)),
                                 ((.006, 0, 0), (0, 0, 0, -1)))
    self.assertAlmostEqual(distance, .006)
    self.assertAlmostEqual(angle, 0.)

  def test_failures_count_and_partial_matrix_not_complete(self):
    rows = [dict(case=spec, success=False, error='failure', timeout=True)
            for spec in cases()[:3]]
    result = summarize(rows)
    self.assertEqual(result['cases'], 3)
    self.assertEqual(result['successes'], 0)
    self.assertEqual(result['errors'], 3)
    self.assertFalse(result['complete_development_matrix'])
    with self.assertRaises(ValueError):
      summarize([])

  def test_real_ravens_trial_restores_step_and_has_no_object_attachment(self):
    original = p.stepSimulation
    task = RetentionTask()
    assets = str(Path(__file__).parent / 'environments/assets')
    env = Environment(assets, task=task)
    try:
      result = run_case(env, task, cases()[2])
      self.assertIs(p.stepSimulation, original)
      self.assertIsNone(result['error'])
      self.assertGreater(len(result['samples']), 10)
      self.assertEqual(len(env.obj_ids['rigid']), 1)
      for index in range(p.getNumConstraints()):
        info = p.getConstraintInfo(p.getConstraintUniqueId(index))
        self.assertNotIn(task.body, (info[0], info[2]))
    finally:
      env.close()


if __name__ == '__main__':
  unittest.main()
