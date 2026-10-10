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
              'cartesian-v1')

  def __init__(self, profile='baseline'):
    if profile not in self.profiles:
      raise ValueError('Unknown manipulation profile: %s' % profile)
    self.profile = profile
    self.last_result = {}

  def __call__(self, movej, movep, ee, pose0, pose1):
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
    if self.profile == 'clearance-v1':
      # Experimental shallower grasp: keep the pads clear of the table for
      # short objects. Uses the observed top only, never a simulator pose.
      pick_z = max(.027, top - .012)
    place_z = support + pick_z + .002
    clearance = max(.30, place_z + .12)
    def pose_at(pose, height):
      return ((pose[0][0], pose[0][1], height), pose[1])
    def move(pose):
      timeout = movep(pose, speed=.001 if self.profile == 'slow-v1' else .002)
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
    def loaded_move(start, end):
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
    if loaded_move(pose_at(pose0, pick_z), pose_at(pose0, clearance)):
      return True
    self.last_result['grasp_success'] = bool(ee.check_grasp())
    if not self.last_result['grasp_success']:
      self.last_result['failure_reason'] = 'lost_grasp_after_lift'
      ee.release()
      return False
    if loaded_move(pose_at(pose0, clearance), pose_at(pose1, clearance)):
      return True
    if self.profile == 'cartesian-v1':
      self.last_result['after_transfer'] = bool(ee.check_grasp())
    if self.profile == 'guarded-v1' and not retained('after_transfer'):
      return False
    if loaded_move(pose_at(pose1, clearance), pose_at(pose1, place_z)):
      return True
    if self.profile == 'cartesian-v1':
      self.last_result['before_release'] = bool(ee.check_grasp())
    if self.profile == 'guarded-v1' and not retained('before_release'):
      if move(pose_at(pose1, clearance)):
        return True
      return False
    ee.release()
    return move(pose_at(pose1, clearance))
