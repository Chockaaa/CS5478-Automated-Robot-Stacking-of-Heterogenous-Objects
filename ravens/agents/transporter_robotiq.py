"""Original Transporter networks with support-height decoding for cubes."""
import numpy as np
from ravens.agents.transporter import TransporterAgent
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.utils import utils


class RobotiqTransporterAgent(TransporterAgent):
  """Predict picks/places from RGB-D only; no oracle or object poses in act."""
  def __init__(self, name, task, root_dir, n_rotations=36):
    super().__init__(name, task, root_dir, n_rotations)
    self.attention = Attention(self.in_shape, 1, utils.preprocess)
    self.transport = Transport(self.in_shape, self.n_rotations,
                                self.crop_size, utils.preprocess)

  def get_sample(self, dataset, augment=True):
    img, p0, theta0, p1, theta1 = super().get_sample(dataset, augment=False)
    if augment:
      # Fixed-yaw cube task: translation augmentation preserves executable
      # jaw orientations; arbitrary image rotation would not.
      img, _, (p0, p1), _ = utils.perturb(img, [p0, p1], set_theta_zero=True)
    return img, p0, theta0, p1, theta1

  def act(self, obs, info=None, goal=None):
    action = super().act(obs, info=None, goal=None)
    image = self.get_image(obs)
    hmap = image[:, :, 3]
    place_pos, place_rot = action['pose1']
    row, col = utils.xyz_to_pix(place_pos, self.bounds, self.pix_size)
    # A pyramid target can lie in the gap between supporting blocks. Use
    # the nearby support top, not the empty centre pixel's table height.
    patch = hmap[max(0, row-3):row+4, max(0, col-3):col+4]
    place_pos[2] = np.max(patch)
    action['pose1'] = (place_pos, place_rot)
    return action
