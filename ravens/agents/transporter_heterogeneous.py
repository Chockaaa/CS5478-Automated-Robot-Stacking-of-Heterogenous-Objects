"""Transporter XY policy with RGB-D-derived upright grasp geometry.

This baseline uses original Attention and Transport ResNets, a smaller workspace
crop at 4 mm/pixel, and one rotation bin. An explicit image-geometry adapter
aligns the parallel jaws; this is not learned rotation prediction or a VLA.
"""
import cv2
import numpy as np
from ravens.agents.transporter import TransporterAgent
from ravens.models.attention import Attention
from ravens.models.transport import Transport
from ravens.tasks.robotiq import AdaptiveRobotiqPickPlace
from ravens.utils import utils


class HeterogeneousTransporterAgent(TransporterAgent):
  decoder_version = 'rgbd-refined-v4-feasible-picks'

  def __init__(self, name, task, root_dir):
    super().__init__(name, task, root_dir, n_rotations=1)
    self.pix_size = .004
    self.bounds = np.array([[.276, .724], [-.388, .380], [0, .28]])
    self.in_shape = (192, 112, 6)
    self.crop_size = 32
    self.geometry_pix_size = .001
    self.attention = Attention(self.in_shape, 1, utils.preprocess)
    self.transport = Transport(self.in_shape, 1, self.crop_size, utils.preprocess)

  def get_sample(self, dataset, augment=True):
    # Fixed tower region: no augmentation translating the task's destination.
    del augment
    img, p0, _, p1, _ = super().get_sample(dataset, augment=False)
    return img, p0, 0., p1, 0.

  def act(self, obs, info=None, goal=None):
    del info, goal
    image = self.get_image(obs)
    hmap = image[:, :, 3]
    low, high = AdaptiveRobotiqPickPlace.pick_height_limits
    valid_pick = (hmap >= low) & (hmap <= high)
    if not np.any(valid_pick):
      return None
    # Mask logits, not probabilities: very unlikely feasible pixels can
    # otherwise all underflow to zero after softmax and lose their ranking.
    pick = np.asarray(self.attention.forward(image, softmax=False)).reshape(
        hmap.shape + (1,))
    # Apply the executor's table-grasp height limits before selecting a pixel.
    # Attention still ranks the candidates; no object IDs or goal-state query.
    pick = np.where(valid_pick[:, :, None], pick, -np.inf)
    pixel = np.unravel_index(np.argmax(pick), pick.shape)[:2]
    place = self.transport.forward(image, pixel)
    target = np.unravel_index(np.argmax(place), place.shape)[:2]
    xyz = np.array(utils.pix_to_xyz(pixel, hmap, self.bounds, self.pix_size))
    dest = np.array(utils.pix_to_xyz(target, hmap, self.bounds, self.pix_size))
    # Refine only the object selected by Attention, from higher-resolution
    # RGB-D geometry. No object IDs, simulator poses or alternative planner.
    geometry = self.get_geometry_heightmap(obs)
    grasp_pixel = utils.xyz_to_pix(xyz, self.bounds, self.geometry_pix_size)
    table = geometry[(geometry > 0) & (geometry < .008)]
    table_height = float(np.median(table)) if table.size else 0.
    mask = np.uint8((geometry > .020) & (np.abs(geometry - xyz[2]) < .006))
    # Fine point-cloud rasters contain empty bins between camera samples.
    # Close small sampling holes, then associate only within 6 mm of Attention.
    mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((3, 3), np.uint8))
    _, labels = cv2.connectedComponents(mask)
    row, col = grasp_pixel
    radius = max(1, int(np.ceil(.006 / self.geometry_pix_size)))
    r0, c0 = max(0, row-radius), max(0, col-radius)
    nearby = np.argwhere(labels[r0:row+radius+1, c0:col+radius+1] > 0)
    label = 0
    if nearby.size:
      nearby += (r0, c0)
      nearest = nearby[np.argmin(np.sum((nearby - grasp_pixel)**2, axis=1))]
      if np.linalg.norm(nearest - grasp_pixel) <= radius:
        label = labels[tuple(nearest)]
    yaw = 0.
    if label:
      points = np.argwhere(labels == label)
      if len(points) >= 6:
        rect = cv2.minAreaRect(points[:, ::-1].astype(np.float32))
        corners = cv2.boxPoints(rect)
        edges = np.roll(corners, -1, axis=0) - corners
        edge = edges[np.argmax(np.linalg.norm(edges, axis=1))]
        yaw = float(np.arctan2(edge[1], edge[0]) + np.pi / 2)
        sides = sorted(rect[1])
        period = np.pi / 2 if sides[1] < 1.15 * max(sides[0], 1) else np.pi
        yaw = (yaw + period/2) % period - period/2
        occupancy = len(points) / max((sides[0]+1)*(sides[1]+1), 1)
        if occupancy < .86 and sides[1] < 1.15 * max(sides[0], 1):
          yaw = 0.  # Approximately circular top: yaw is unconstrained.
        xyz[:2] = self.bounds[:2, 0] + (np.array(rect[0]) + .5) * self.geometry_pix_size
        heights = geometry[(labels == label) & (geometry > .020)]
        xyz[2] = max(0., float(np.median(heights)) - table_height)
    target_pixel = utils.xyz_to_pix(dest, self.bounds, self.geometry_pix_size)
    row, col = target_pixel
    support = geometry[max(0, row-2):row+3, max(0, col-2):col+3]
    support = support[support > 0]
    dest[2] = max(0., float(np.median(support)) - table_height) if support.size else 0.
    quat = utils.eulerXYZ_to_quatXYZW((0, 0, yaw))
    return dict(pose0=(xyz, quat), pose1=(dest, quat))

  def get_geometry_heightmap(self, obs):
    _, heightmap = utils.get_fused_heightmap(
        obs, self.cam_config, self.bounds, self.geometry_pix_size)
    return heightmap
