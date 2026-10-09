"""Robotiq 2F-85: coupled fingers and contact-based physical grasping."""
import os
import numpy as np
import pybullet as p


class Robotiq85:
  """Bundled Robotiq meshes; no object attachment or pose snapping."""
  # Ravens' UR5 IK marker is 120 mm from the flange. Closed pads centre
  # near 140 mm. movep compensates for that extra 20 mm along world Z.
  tip_offset = .020
  joint_tolerance = .001

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
      p.changeDynamics(self.body, link, lateralFriction=1,
                       spinningFriction=.001, restitution=0)
      for robot_link in range(-1, p.getNumJoints(robot)):
        p.setCollisionFilterPair(robot, self.body, robot_link, link, 0)
    self.release()

  def step(self):
    pass

  def _command(self, angle):
    p.setJointMotorControl2(self.body, self.driver, p.POSITION_CONTROL,
                           targetPosition=angle, maxVelocity=.5,
                           force=2, positionGain=.1)

  def _contacts(self):
    contacts = [{pt[2] for pt in p.getContactPoints(
        bodyA=self.body, linkIndexA=pad)
                 if pt[2] in self.obj_ids['rigid'] and pt[9] > .01}
                for pad in self.pads]
    return contacts[0] & contacts[1]

  def activate(self):
    self.activated = True
    self._command(.8)
    for _ in range(240):
      p.stepSimulation()
    objects = self._contacts()
    self.grasped = next(iter(objects)) if len(objects) == 1 else None

  def release(self):
    self.activated = False
    self.grasped = None
    self._command(0)
    for _ in range(240):
      p.stepSimulation()

  def check_grasp(self):
    return self.grasped is not None and self.grasped in self._contacts()

  def detect_contact(self):
    return self.grasped is not None and any(
        pt[2] not in (self.body, self.robot)
        for pt in p.getContactPoints(bodyA=self.grasped))


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
