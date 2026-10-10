"""Scene sampling and real PyBullet reset tests (not manipulation tests)."""
from pathlib import Path
import unittest
import numpy as np
import pybullet as p
from ravens.environments.environment import Environment
from ravens.tasks.stack_heterogeneous import sample_scene, StackHeterogeneous


class SceneTest(unittest.TestCase):
  def test_reproducibility_and_split_streams(self):
    first = sample_scene(17)
    np.random.seed(999)
    np.random.uniform(size=100)
    self.assertEqual(first, sample_scene(17))
    self.assertNotEqual(first['objects'], sample_scene(18)['objects'])
    for split in ('validation', 'test'):
      self.assertNotEqual(first['objects'], sample_scene(17, split)['objects'])

  def test_invalid_inputs(self):
    for seed, split in ((-1, 'train'), (1.5, 'train'), (0, 'unknown')):
      with self.assertRaises(ValueError):
        sample_scene(seed, split)

  def test_geometry_and_separation(self):
    for split in ('train', 'validation', 'test'):
      for seed in range(100):
        objects = sample_scene(seed, split)['objects']
        self.assertEqual({o['family'] for o in objects},
                         {'cube', 'cuboid', 'cylinder'})
        for i, obj in enumerate(objects):
          self.assertTrue(all(.028 <= d <= .060 for d in obj['size']))
          self.assertGreater(obj['mass'], 0)
          self.assertGreater(obj['position'][0] - obj['footprint_radius'], .25)
          self.assertLess(obj['position'][0] + obj['footprint_radius'], .75)
          for other in objects[:i]:
            distance = np.linalg.norm(np.array(obj['position'][:2]) -
                                      other['position'][:2])
            self.assertGreater(distance, obj['footprint_radius'] +
                               other['footprint_radius'] + .035)

  def test_environment_reset_and_empty_tower_score(self):
    assets = Path(__file__).resolve().parents[1] / 'environments' / 'assets'
    task = StackHeterogeneous(seed=9, split='test')
    env = Environment(str(assets), task=task)
    try:
      env.seed(9)
      obs = env.reset()
      self.assertIn('color', obs)
      self.assertIn('depth', obs)
      self.assertEqual(len(env.obj_ids['rigid']), 3)
      self.assertEqual(task.manifest['schema_version'], 2)
      self.assertFalse(p.getCollisionShapeData(task.target_marker, -1))
      for body, obj in zip(task.object_ids, task.manifest['objects']):
        self.assertAlmostEqual(p.getBasePositionAndOrientation(body)[0][2],
                               obj['size'][2]/2 - .001, delta=.001)
      self.assertFalse(task.done())
      for body, obj in zip(task.object_ids, task.manifest['objects']):
        self.assertAlmostEqual(p.getDynamicsInfo(body, -1)[0], obj['mass'])
        self.assertEqual(p.getVisualShapeData(body)[0][2],
                         p.GEOM_CYLINDER if obj['family'] == 'cylinder' else p.GEOM_BOX)
      self.assertEqual(task.evaluate()['stable_objects'], 0)
      action = task.oracle(env).act(obs, None)
      self.assertEqual(set(action), {'pose0', 'pose1'})
      self.assertEqual(task.reward()[0], 0.)
    finally:
      env.close()

  def test_scoring_requires_vertical_support_and_rejects_collapse(self):
    assets = Path(__file__).resolve().parents[1] / 'environments' / 'assets'
    task = StackHeterogeneous(seed=0)
    env = Environment(str(assets), task=task)
    try:
      env.seed(0)
      env.reset()
      # Synthetic evaluator fixture only: execution never teleports objects.
      order = sorted(range(3), key=lambda i: (
          task.manifest['objects'][i]['family'] != 'cylinder',
          np.prod(task.manifest['objects'][i]['size'][:2])), reverse=True)
      height = -.001
      for i in order:
        half = task.manifest['objects'][i]['size'][2] / 2
        p.resetBasePositionAndOrientation(task.object_ids[i],
                                         [.5, .18, height + half], [0, 0, 0, 1])
        p.resetBaseVelocity(task.object_ids[i], [0, 0, 0], [0, 0, 0])
        height += half * 2
      for _ in range(480):
        p.stepSimulation()
      self.assertTrue(task.evaluate()['success'])
      top = task.object_ids[order[-1]]
      p.resetBasePositionAndOrientation(top, [.6, .18, .1], [0, 0, 0, 1])
      for _ in range(480):
        p.stepSimulation()
      self.assertFalse(task.evaluate()['success'])
      self.assertEqual(task.evaluate()['stable_objects'], 2)
    finally:
      env.close()


if __name__ == '__main__':
  unittest.main()
