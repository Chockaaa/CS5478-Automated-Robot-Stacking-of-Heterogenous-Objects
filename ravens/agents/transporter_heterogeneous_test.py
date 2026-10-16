"""Test observation-only decoding without constructing neural weights."""
import unittest
import numpy as np
from ravens.agents.transporter_heterogeneous import HeterogeneousTransporterAgent


class FixedScores:
  def __init__(self, pixel):
    self.pixel = pixel

  def forward(self, image, *args, **kwargs):
    del args, kwargs
    scores = np.zeros(image.shape[:2] + (1,), dtype=np.float32)
    scores[self.pixel + (0,)] = 1
    return scores


class AgentDecodeTest(unittest.TestCase):
  def test_rgbd_geometry_and_ignored_privileged_info(self):
    agent = HeterogeneousTransporterAgent.__new__(HeterogeneousTransporterAgent)
    agent.bounds = np.array([[.276, .724], [-.388, .252], [0, .28]])
    agent.pix_size = .004
    agent.geometry_pix_size = .004
    image = np.zeros((160, 112, 6), dtype=np.float32)
    image[20:28, 30:44, 3:] = .04  # A cuboid's observed top face.
    image[140:147, 50:58, 3:] = .08  # Existing support surface.
    agent.get_image = lambda obs: image.copy()
    agent.get_geometry_heightmap = lambda obs: image[:, :, 3].copy()
    agent.attention = FixedScores((24, 37))
    agent.transport = FixedScores((143, 54))
    first = agent.act({}, info={'object_pose': 'must not be read'})
    second = agent.act({}, info=None)
    for key in ('pose0', 'pose1'):
      np.testing.assert_allclose(first[key][0], second[key][0])
      np.testing.assert_allclose(first[key][1], second[key][1])
    self.assertAlmostEqual(first['pose0'][0][2], .04)
    self.assertAlmostEqual(first['grasp_height'], .04)
    self.assertAlmostEqual(first['grasp_width'], .032)
    self.assertAlmostEqual(first['pose1'][0][2], .08)
    np.testing.assert_allclose(first['pose0'][1], first['pose1'][1])
    # Jaw direction is the rectangle's short axis, modulo gripper symmetry.
    self.assertAlmostEqual(abs(first['pose0'][1][2]), np.sqrt(.5), places=5)
    # Even a higher score on the tall stack cannot bypass table-grasp limits.
    agent.attention = FixedScores((143, 54))
    feasible = agent.act({})
    self.assertAlmostEqual(feasible['pose0'][0][2], .04)

  def test_no_feasible_height_stops_without_model_or_oracle(self):
    agent = HeterogeneousTransporterAgent.__new__(HeterogeneousTransporterAgent)
    image = np.zeros((192, 112, 6), dtype=np.float32)
    image[80:90, 30:40, 3:] = .10
    agent.get_image = lambda obs: image
    self.assertIsNone(agent.act({}, info={'hidden_pose': 'not used'}))

  def test_sampling_hole_does_not_reset_grasp_rotation(self):
    agent = HeterogeneousTransporterAgent.__new__(HeterogeneousTransporterAgent)
    agent.bounds = np.array([[.276, .724], [-.388, .380], [0, .28]])
    agent.pix_size = .004
    agent.geometry_pix_size = .001
    image = np.zeros((192, 112, 6), dtype=np.float32)
    image[24, 37, 3:] = .04
    geometry = np.zeros((768, 448), dtype=np.float32)
    geometry[80:112, 120:176] = .04
    geometry[96, 148] = 0  # Empty raster bin exactly at the selected pick.
    geometry[570:578, 214:222] = .08
    geometry[572, 216] = 0
    agent.get_image = lambda obs: image.copy()
    agent.get_geometry_heightmap = lambda obs: geometry.copy()
    agent.attention = FixedScores((24, 37))
    agent.transport = FixedScores((143, 54))
    action = agent.act({})
    np.testing.assert_allclose(action['pose0'][0], [.424, -.292, .04], atol=1e-6)
    self.assertAlmostEqual(abs(action['pose0'][1][2]), np.sqrt(.5), places=5)
    self.assertAlmostEqual(action['pose1'][0][2], .08)
    self.assertAlmostEqual(action['grasp_height'], .04)
    self.assertAlmostEqual(action['grasp_width'], .032)
    # Placement XY remains the network's choice, not a fixed goal override.
    np.testing.assert_allclose(action['pose1'][0][:2], [.492, .184])


if __name__ == '__main__':
  unittest.main()
