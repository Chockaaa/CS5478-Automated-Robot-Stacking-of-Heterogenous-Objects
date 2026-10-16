"""Robotiq 2F-85: coupled fingers and contact-based physical grasping."""
import os
import math
import pybullet as p


class Robotiq85:
  """Bundled Robotiq meshes; no object attachment or pose snapping."""
  # Ravens' UR5 IK marker is 120 mm from the flange. Closed pads centre
  # near 140 mm. movep compensates for that extra 20 mm along world Z.
  tip_offset = .020
  joint_tolerance = .001
  contact_preload = .02  # Additional driver radians after bilateral contact.
  grip_force = 2  # Driver motor force limit; overridden only in experiments.
  pad_friction = 1.  # Lateral friction of gripper links; experiments only.
  pad_spinning_friction = .001

  def __init__(self, assets_root, robot, ee, obj_ids):
    self.robot, self.obj_ids = robot, obj_ids
    self.activated = False
    self.grasped = None
    pose = (p.getBasePositionAndOrientation(robot) if ee == -1 else
            p.getLinkState(robot, ee))
    self.body = p.loadURDF(os.path.join(
        assets_root, 'ur5/gripper/robotiq-stacking.urdf'), pose[0], pose[1])
    mount = p.createConstraint(robot, ee, self.body, -1, p.JOINT_FIXED,
                               (0, 0, 0), (0, 0, 0), (0, 0, 0))
    p.changeConstraint(mount, maxForce=500)
    names = {p.getJointInfo(self.body, i)[1].decode(): i
             for i in range(p.getNumJoints(self.body))}
    prefix = 'robotiq_2f_85_'
    self.driver = names[prefix + 'right_driver_joint']
    self.pads = [names[prefix + side + '_pad_joint']
                 for side in ('right', 'left')]
    for suffix, multiplier in (
        ('right_follower_joint', -1), ('right_spring_link_joint', 1),
        ('left_driver_joint', 1), ('left_follower_joint', -1),
        ('left_spring_link_joint', 1)):
      slave = names[prefix + suffix]
      p.setJointMotorControl2(self.body, slave, p.VELOCITY_CONTROL, force=0)
      gear = p.createConstraint(self.body, self.driver, self.body, slave,
                                p.JOINT_GEAR, (1, 0, 0), (0, 0, 0), (0, 0, 0))
      p.changeConstraint(gear, gearRatio=-multiplier, erp=.1, maxForce=100)
    for link in range(-1, p.getNumJoints(self.body)):
      p.changeDynamics(self.body, link, lateralFriction=self.pad_friction,
                       spinningFriction=self.pad_spinning_friction, restitution=0)
      for robot_link in range(-1, p.getNumJoints(robot)):
        p.setCollisionFilterPair(robot, self.body, robot_link, link, 0)
    self.release()

  def step(self):
    pass

  def _command(self, angle):
    p.setJointMotorControl2(self.body, self.driver, p.POSITION_CONTROL,
                           targetPosition=angle, maxVelocity=.5,
                           force=self.grip_force, positionGain=.1)

  def configure_grasp(self, width=None):
    """Configure an observed grasp without changing the baseline controller."""
    del width

  def _contacts(self):
    contacts = [{pt[2] for pt in p.getContactPoints(
        bodyA=self.body, linkIndexA=pad)
                 if pt[2] in self.obj_ids['rigid'] and pt[9] > .01}
                for pad in self.pads]
    return contacts[0] & contacts[1]

  def activate(self):
    self.activated = True
    self._command(.8)
    # 0.8 rad at the 0.5 rad/s motor limit needs at least 1.6 seconds.
    # Starting the lift after one second leaves narrow objects ungrasped.
    for _ in range(480):
      p.stepSimulation()
      objects = self._contacts()
      if len(objects) == 1:
        # Hold the achieved opening once both pads touch. Continuing to drive
        # toward fully closed can rotate lightweight objects during transport.
        angle = p.getJointState(self.body, self.driver)[0]
        self._command(min(.8, angle + self.contact_preload))
        for _ in range(120):
          p.stepSimulation()
        break
    objects = self._contacts()
    self.grasped = next(iter(objects)) if len(objects) == 1 else None

  def release(self):
    self.activated = False
    self.grasped = None
    self._command(0)
    # Fully open before retreating, including after a narrow-object grasp.
    for _ in range(480):
      p.stepSimulation()

  def check_grasp(self):
    return self.grasped is not None and self.grasped in self._contacts()

  def detect_contact(self):
    return self.grasped is not None and any(
        pt[2] not in (self.body, self.robot)
        for pt in p.getContactPoints(bodyA=self.grasped))


class RetentionRobotiq85(Robotiq85):
  """Experimental intermediate contact preload; not the default gripper."""
  contact_preload = .03


class SizeAwareRobotiq85(Robotiq85):
  """Experimental bounded force schedule from observed jaw closing width."""
  min_grip_force = 2.
  max_grip_force = 3.

  def configure_grasp(self, width=None):
    if width is None or not math.isfinite(width) or not .02 <= width <= .06:
      raise ValueError('size-aware grasp requires an observed width in [20, 60] mm')
    # Retain the baseline force through 40 mm, then linearly add at most 1 N.
    self.grip_force = min(self.max_grip_force, self.min_grip_force +
                          max(0., width - .04) / .01)


class SizeAwareV2Robotiq85(SizeAwareRobotiq85):
  """Narrow force increase for the wide-object failure regime only."""
  def configure_grasp(self, width=None):
    if width is None or not math.isfinite(width) or not .02 <= width <= .06:
      raise ValueError('size-aware grasp requires an observed width in [20, 60] mm')
    self.grip_force = min(self.max_grip_force, self.min_grip_force +
                          max(0., width - .048) / .002)


class RobotiqPickPlace:
  """Surface pick + support-surface place, matching RGB-D agent outputs.

  This initial task uses 40 mm cubes. No simulator object pose is queried
  to correct a learned action. Failed picks terminate the primitive safely.
  """
  def __call__(self, movej, movep, ee, pose0, pose1):
    del movej
    def at_height(pose, z):
      return ((pose[0][0], pose[0][1], z), pose[1])
    pick = at_height(pose0, max(.021, pose0[0][2] - .017))
    place = at_height(pose1, max(.022, pose1[0][2] + .023))
    ee.release()
    for target in (at_height(pick, .32), pick):
      if movep(target, speed=.003):
        ee.release()
        return True
    for _ in range(60):
      p.stepSimulation()
    ee.activate()
    if movep(at_height(pick, .32), speed=.002):
      ee.release()
      return True
    if not ee.check_grasp():
      ee.release()
      return False
    for target in (at_height(place, .32), place):
      if movep(target, speed=.002):
        ee.release()
        return True
    ee.release()
    return movep(at_height(place, .32), speed=.003)


class AdaptiveRobotiqPickPlace:
  """Upright table-object grasp using observed top and support heights.

  Objects are never attached or teleported. Full-open fingers close until
  contact; initial objects must be separated and on the table. Orientation
  comes from the action, not simulator geometry.
  """
  pick_height_limits = (.025, .065)

  profiles = ('baseline', 'clearance-v1', 'guarded-v1', 'retention-v1', 'slow-v1',
              'cartesian-v1', 'size-aware-v1', 'size-aware-v2', 'size-aware-v3',
              'size-aware-v4', 'monitored-v1', 'placement-v1', 'accel-v1',
              'cartesian-accel-v2')

  def __init__(self, profile='baseline'):
    if profile not in self.profiles:
      raise ValueError('Unknown manipulation profile: %s' % profile)
    self.profile = profile
    self.last_result = {}

  @staticmethod
  def size_aware_pick_height(height, width):
    if (not math.isfinite(height) or not math.isfinite(width) or
        not .025 <= height <= .065 or not .02 <= width <= .06):
      raise ValueError('invalid observed grasp geometry')
    # The pad bottom is about 22 mm below the commanded TCP in this model.
    # A 24 mm floor keeps it clear of the table; otherwise target the object's
    # vertical centre instead of a fixed distance below its top surface.
    return max(.024, height / 2)

  @staticmethod
  def size_aware_v2_pick_height(height, width):
    if (not math.isfinite(height) or not math.isfinite(width) or
        not .025 <= height <= .065 or not .02 <= width <= .06):
      raise ValueError('invalid observed grasp geometry')
    if height <= .035:
      return max(.022, height - .016)  # Clear table without v1's 5 mm jump.
    if height >= .05 and width >= .048:
      return max(.024, height / 2)  # Centre only in the wide/tall regime.
    return max(.019, height - .016)  # Preserve the successful baseline depth.

  @staticmethod
  def size_aware_v3_pick_height(height, width):
    """Rejected lower-grasp ablation retained for result reproducibility."""
    if (not math.isfinite(height) or not math.isfinite(width) or
        not .025 <= height <= .065 or not .02 <= width <= .06):
      raise ValueError('invalid observed grasp geometry')
    if height >= .05 and width >= .048:
      return max(.024, height / 2)
    return max(.019, height - .017)

  @staticmethod
  def size_aware_v4_pick_height(height, width):
    """Use baseline depth normally and centre wide/tall grasps."""
    if (not math.isfinite(height) or not math.isfinite(width) or
        not .025 <= height <= .065 or not .02 <= width <= .06):
      raise ValueError('invalid observed grasp geometry')
    if height >= .05 and width >= .048:
      return max(.024, height / 2)
    # Initial object centres sit about 1 mm over the plane, so this reproduces
    # baseline top-minus-17-mm grasping from an estimated object height.
    return max(.019, height - .016)

  def __call__(self, movej, movep, ee, pose0, pose1,
               grasp_height=None, grasp_width=None):
    del movej
    self.last_result = dict(grasp_success=False, timeout=False)
    top, support = float(pose0[0][2]), float(pose1[0][2])
    if not (self.pick_height_limits[0] <= top <= self.pick_height_limits[1]
            and 0 <= support <= .20):
      self.last_result['invalid_height'] = True
      return False
    if (self.profile == 'cartesian-v1' and
        abs(sum(a*b for a, b in zip(pose0[1], pose1[1]))) < 1 - 1e-6):
      self.last_result['failure_reason'] = 'unsupported_loaded_rotation'
      return False
    pick_z = max(.019, top - .017)
    if self.profile in ('size-aware-v1', 'size-aware-v2', 'size-aware-v3',
                        'size-aware-v4'):
      try:
        geometry = ({
            'size-aware-v1': self.size_aware_pick_height,
            'size-aware-v2': self.size_aware_v2_pick_height,
            'size-aware-v3': self.size_aware_v3_pick_height,
            'size-aware-v4': self.size_aware_v4_pick_height,
        }[self.profile])
        pick_z = geometry(float(grasp_height), float(grasp_width))
        ee.configure_grasp(float(grasp_width))
      except (TypeError, ValueError):
        self.last_result['invalid_geometry'] = True
        return False
      self.last_result.update(grasp_height=float(grasp_height),
                              grasp_width=float(grasp_width), pick_z=pick_z,
                              grip_force=ee.grip_force)
    if self.profile == 'clearance-v1':
      # Experimental shallower grasp: keep the pads clear of the table for
      # short objects. Uses the observed top only, never a simulator pose.
      pick_z = max(.027, top - .012)
    place_z = support + pick_z + .002
    clearance = max(.30, place_z + .12)
    def pose_at(pose, height):
      return ((pose[0][0], pose[0][1], height), pose[1])
    monitor_misses = 0
    monitor_stage = None
    monitor_lost = False
    def monitor_contact():
      nonlocal monitor_misses, monitor_lost
      if ee.check_grasp():
        monitor_misses = 0
        return True
      monitor_misses += 1
      if monitor_misses < 12:
        return True
      monitor_lost = True
      self.last_result['continuous_monitor'] = True
      self.last_result['monitor_miss_steps'] = monitor_misses
      self.last_result['failure_reason'] = 'lost_grasp_during_' + monitor_stage
      return False
    def move(pose, monitor=None, acceleration=None, cartesian_limits=None):
      kwargs = dict(speed=.001 if self.profile == 'slow-v1' else .002)
      if monitor is not None:
        kwargs['monitor'] = monitor
      if acceleration is not None:
        kwargs['acceleration'] = acceleration
      if cartesian_limits is not None:
        kwargs['cartesian_speed'], kwargs['cartesian_acceleration'] = cartesian_limits
      timeout = movep(pose, **kwargs)
      if timeout:
        self.last_result['timeout'] = True
        ee.release()
      return timeout
    def retained(stage):
      result = bool(ee.check_grasp())
      self.last_result[stage] = result
      if not result:
        self.last_result['failure_reason'] = 'lost_grasp_' + stage
        ee.release()
      return result
    def loaded_move(start, end, stage):
      nonlocal monitor_misses, monitor_stage
      monitor_misses, monitor_stage = 0, stage
      if self.profile in ('monitored-v1', 'placement-v1', 'accel-v1',
                          'cartesian-accel-v2'):
        if self.profile == 'placement-v1' and stage == 'lower':
          distance = abs(end[0][2] - start[0][2])
          count = max(1, math.ceil(distance / .01))
          for index in range(1, count + 1):
            fraction = index / count
            xyz = tuple(a + fraction*(b-a) for a, b in zip(start[0], end[0]))
            timeout = move((xyz, end[1]), monitor=monitor_contact)
            if timeout or monitor_lost:
              if monitor_lost:
                ee.release()
              return timeout
          return False
        timeout = move(
            end, monitor=monitor_contact,
            acceleration=.00005 if self.profile == 'accel-v1' else None,
            cartesian_limits=(.08, .4)
            if self.profile == 'cartesian-accel-v2' else None)
        if monitor_lost:
          ee.release()
        return timeout
      if self.profile != 'cartesian-v1':
        return move(end)
      # Constrain the loaded path with <=10 mm Cartesian waypoints rather
      # than one long joint-space interpolation. No object pose is queried.
      # Mixed-tower actions preserve grasp yaw; do not interpolate arbitrary
      # rotations here or silently substitute a different placement yaw.
      distance = math.sqrt(sum((b-a)**2 for a, b in zip(start[0], end[0])))
      count = max(1, math.ceil(distance / .01))
      for index in range(1, count + 1):
        fraction = index / count
        xyz = tuple(a + fraction*(b-a) for a, b in zip(start[0], end[0]))
        if move((xyz, end[1])):
          return True
      return False
    ee.release()
    if move(pose_at(pose0, clearance)) or move(pose_at(pose0, pick_z)):
      return True
    for _ in range(60):
      p.stepSimulation()
    ee.activate()
    if self.profile == 'guarded-v1':
      # Verify contact before lifting and after a short lift. Failures return
      # to the runner for a fresh RGB-D observation and a budgeted new action;
      # no hidden retries, object-pose correction or oracle fallback.
      if not retained('after_close'):
        return False
      if move(pose_at(pose0, pick_z + .04)):
        return True
      if not retained('after_short_lift'):
        return False
    if loaded_move(pose_at(pose0, pick_z), pose_at(pose0, clearance), 'lift'):
      return True
    if monitor_lost:
      return False
    self.last_result['grasp_success'] = bool(ee.check_grasp())
    if not self.last_result['grasp_success']:
      self.last_result['failure_reason'] = 'lost_grasp_after_lift'
      ee.release()
      return False
    if loaded_move(pose_at(pose0, clearance), pose_at(pose1, clearance),
                   'transfer'):
      return True
    if monitor_lost:
      return False
    if self.profile == 'cartesian-v1':
      self.last_result['after_transfer'] = bool(ee.check_grasp())
    if self.profile == 'guarded-v1' and not retained('after_transfer'):
      return False
    if loaded_move(pose_at(pose1, clearance), pose_at(pose1, place_z),
                   'lower'):
      return True
    if monitor_lost:
      return False
    if self.profile == 'cartesian-v1':
      self.last_result['before_release'] = bool(ee.check_grasp())
    if self.profile == 'guarded-v1' and not retained('before_release'):
      if move(pose_at(pose1, clearance)):
        return True
      return False
    if self.profile == 'placement-v1':
      monitor_misses, monitor_stage = 0, 'pre_release_hold'
      for _ in range(120):
        p.stepSimulation()
        if not monitor_contact():
          ee.release()
          return False
      self.last_result['pre_release_hold'] = True
    ee.release()
    return move(pose_at(pose1, clearance))
