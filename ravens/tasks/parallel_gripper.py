"""Physical, motor-driven parallel fingers with optional contact assistance."""
import os
import numpy as np
import pybullet as p


class ParallelJawGripper:
  """Two sliding fingers. Assistance is only enabled after bilateral contact."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    self.robot, self.ee, self.obj_ids = robot, ee, obj_ids
    p.setPhysicsEngineParameter(numSolverIterations=100)
    self.activated = False
    self.contact_constraint = None
    self.grasped_object = None
    self.assisted = False
    self.joint_tolerance = .001
    self.contact_history = []
    self.body = p.loadURDF(os.path.join(
        assets_root, 'ur5/gripper/parallel-jaw.urdf'))
    self.mount = p.createConstraint(
        robot, ee, self.body, -1, p.JOINT_FIXED, (0, 0, 0),
        (0, 0, 0), (0, 0, 0))
    p.changeConstraint(self.mount, maxForce=500)
    for link in range(-1, 2):
      p.changeDynamics(self.body, link, lateralFriction=1.5,
                       spinningFriction=.001, restitution=0)
      for robot_link in range(-1, p.getNumJoints(robot)):
        p.setCollisionFilterPair(robot, self.body, robot_link, link, 0)
    self.release()

  def _command(self, width):
    for joint in (0, 1):
      p.setJointMotorControl2(self.body, joint, p.POSITION_CONTROL,
                             targetPosition=width / 2, force=2,
                             maxVelocity=.08, positionGain=.2)

  def _contacts(self):
    sides = []
    for joint in (0, 1):
      sides.append({pt[2] for pt in p.getContactPoints(
          bodyA=self.body, linkIndexA=joint)
                    if pt[2] in self.obj_ids['rigid'] and pt[9] > .01})
    return sides[0] & sides[1]

  def activate(self):
    if self.activated:
      return
    self.activated = True
    self._command(0)
    for _ in range(180):
      p.stepSimulation()
    contacts = self._contacts()
    self.contact_history.append(bool(contacts))
    if len(contacts) != 1:
      return
    self.grasped_object = contacts.pop()
    if self.assisted:
      pose = p.getBasePositionAndOrientation(self.body)
      inv = p.invertTransform(*pose)
      rel = p.multiplyTransforms(*inv,
          *p.getBasePositionAndOrientation(self.grasped_object))
      self.contact_constraint = p.createConstraint(
          self.body, -1, self.grasped_object, -1, p.JOINT_FIXED,
          (0, 0, 0), rel[0], (0, 0, 0),
          parentFrameOrientation=rel[1])

  def check_grasp(self):
    return (self.grasped_object is not None and
            (self.contact_constraint is not None or
             self.grasped_object in self._contacts()))

  def release(self):
    if self.contact_constraint is not None:
      p.removeConstraint(self.contact_constraint)
    self.contact_constraint = None
    self.grasped_object = None
    self.activated = False
    self._command(.09)
    for _ in range(120):
      p.stepSimulation()

  def detect_contact(self):
    if self.grasped_object is None:
      return False
    return any(pt[2] not in (self.body, self.robot)
               for pt in p.getContactPoints(bodyA=self.grasped_object))


class ParallelPickPlace:
  """Scripted centre grasp and absolute object placement; simulator feedback."""

  def __call__(self, movej, movep, ee, pose0, pose1):
    del movej
    def above(pose):
      return ((pose[0][0], pose[0][1], .32), pose[1])
    ee.release()
    for pose in (above(pose0), pose0):
      if movep(pose, speed=.005):
        ee.release()
        return True
    for _ in range(60):
      p.stepSimulation()
    ee.activate()
    if movep(above(pose0), speed=.003):
      ee.release()
      return True
    if not ee.check_grasp():
      print('Grasp lost during lift', flush=True)
      ee.release()
      return False
    # These scripted actions specify OBJECT target poses. Measure the grasp
    # transform after lifting to account for movement during finger closure.
    # This is privileged simulator feedback, not a learned vision controller.
    tool = p.getLinkState(ee.robot, 10)
    relative = p.multiplyTransforms(
        *p.invertTransform(tool[0], tool[1]),
        *p.getBasePositionAndOrientation(ee.grasped_object))
    pose1 = p.multiplyTransforms(*pose1, *p.invertTransform(*relative))
    if movep(above(pose1), speed=.005):
      ee.release()
      return True
    for _ in range(60):
      p.stepSimulation()
    # Place at the requested centre height with a 2 mm release clearance.
    target = (np.array(pose1[0]) + (0, 0, .002), pose1[1])
    if movep(target, speed=.003):
      ee.release()
      return True
    ee.release()
    return movep(above(pose1), speed=.003)
