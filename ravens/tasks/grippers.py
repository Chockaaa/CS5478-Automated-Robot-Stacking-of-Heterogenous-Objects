# coding=utf-8
# Copyright 2024 The Ravens Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Classes to handle gripper dynamics."""

import os

import numpy as np
from ravens.utils import pybullet_utils

import pybullet as p

SPATULA_BASE_URDF = 'ur5/spatula/spatula-base.urdf'
SUCTION_BASE_URDF = 'ur5/suction/suction-base.urdf'
SUCTION_HEAD_URDF = 'ur5/suction/suction-head.urdf'
PARALLEL_JAW_URDF = 'ur5/gripper/robotiq_2f_85.urdf'


class Gripper:
  """Base gripper class."""

  def __init__(self, assets_root):
    self.assets_root = assets_root
    self.activated = False

  def step(self):
    """This function can be used to create gripper-specific behaviors."""
    return

  def activate(self, objects):
    del objects
    return

  def release(self):
    return


class Spatula(Gripper):
  """Simulate simple spatula for pushing."""

  def __init__(self, assets_root, robot, ee, obj_ids):  # pylint: disable=unused-argument
    """Creates spatula and 'attaches' it to the robot."""
    super().__init__(assets_root)

    # Load spatula model.
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SPATULA_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))


class ParallelJawGripper(Gripper):
  """Robotiq 2F-85 parallel gripper with a constraint-assisted grasp.

  The fingers are animated using the mimic-joint values from the URDF. Ravens'
  original pick-place primitive was designed for suction and only supplies a
  top-down pick point, not a force-closure trajectory. To keep that action API
  (and Transporter models) unchanged, a valid object between the fingers is
  attached with a fixed constraint after the fingers close.
  """

  def __init__(self, assets_root, robot, ee, obj_ids):
    super().__init__(assets_root)
    self.robot = robot
    self.ee = ee
    self.obj_ids = obj_ids
    self.contact_constraint = None
    self.grasped_object = None
    self.max_grasp_width = 0.085
    self.max_center_offset = 0.03

    # Load at the current tool pose before fixing the gripper to the UR5.
    link_pose = p.getLinkState(robot, ee)
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, PARALLEL_JAW_URDF),
        link_pose[0], link_pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0))

    self.joints = {}
    for joint_id in range(p.getNumJoints(self.body)):
      name = p.getJointInfo(self.body, joint_id)[1].decode('utf-8')
      self.joints[name] = joint_id
    self._set_fingers(0.0)

  def _set_fingers(self, position):
    """Set the driver and PyBullet-unsupported mimic joints immediately."""
    position = float(np.clip(position, 0.0, 0.834))
    targets = {
        'robotiq_2f_85_right_driver_joint': position,
        'robotiq_2f_85_right_follower_joint': -position,
        'robotiq_2f_85_right_spring_link_joint': position,
        'robotiq_2f_85_left_driver_joint': position,
        'robotiq_2f_85_left_follower_joint': -position,
        'robotiq_2f_85_left_spring_link_joint': position,
    }
    for name, target in targets.items():
      joint_id = self.joints.get(name)
      if joint_id is not None:
        p.resetJointState(self.body, joint_id, target)
        p.setJointMotorControl2(
            self.body, joint_id, p.POSITION_CONTROL,
            targetPosition=target, force=100)

  def _tool_pose(self):
    """Return the pose used by Ravens inverse kinematics for pick/place."""
    # In the bundled UR5, tool_tip immediately follows the attachment link.
    tip_link = self.ee + 1
    if tip_link < p.getNumJoints(self.robot):
      state = p.getLinkState(self.robot, tip_link)
      return state[0], state[1]
    state = p.getLinkState(self.robot, self.ee)
    return state[0], state[1]

  def _candidate(self):
    """Find the closest graspable rigid body centered under the tool."""
    tool_pos, tool_rot = self._tool_pose()
    world_to_tool = p.invertTransform(tool_pos, tool_rot)
    candidates = []
    for obj_id in self.obj_ids['rigid']:
      obj_pos, obj_rot = p.getBasePositionAndOrientation(obj_id)
      local_pos, _ = p.multiplyTransforms(
          world_to_tool[0], world_to_tool[1], obj_pos, obj_rot)
      aabb = p.getAABB(obj_id)
      size = np.subtract(aabb[1], aabb[0])
      horizontal_offset = np.linalg.norm(local_pos[:2])
      vertical_offset = abs(tool_pos[2] - aabb[1][2])
      if (horizontal_offset <= self.max_center_offset and
          min(size[0], size[1]) <= self.max_grasp_width and
          vertical_offset <= 0.025):
        candidates.append((horizontal_offset + vertical_offset, obj_id))
    return min(candidates)[1] if candidates else None

  def activate(self):
    """Close the fingers and attach a centered rigid object, if present."""
    if self.activated:
      return
    self._set_fingers(0.72)
    obj_id = self._candidate()
    if obj_id is not None:
      body_pose = p.getBasePositionAndOrientation(self.body)
      obj_pose = p.getBasePositionAndOrientation(obj_id)
      world_to_body = p.invertTransform(body_pose[0], body_pose[1])
      obj_in_body = p.multiplyTransforms(
          world_to_body[0], world_to_body[1], obj_pose[0], obj_pose[1])
      self.contact_constraint = p.createConstraint(
          parentBodyUniqueId=self.body,
          parentLinkIndex=-1,
          childBodyUniqueId=obj_id,
          childLinkIndex=-1,
          jointType=p.JOINT_FIXED,
          jointAxis=(0, 0, 0),
          parentFramePosition=obj_in_body[0],
          parentFrameOrientation=obj_in_body[1],
          childFramePosition=(0, 0, 0),
          childFrameOrientation=(0, 0, 0, 1))
      self.grasped_object = obj_id
    self.activated = True

  def release(self):
    """Detach the held object and open the fingers."""
    if self.contact_constraint is not None:
      try:
        p.removeConstraint(self.contact_constraint)
      except p.error:
        pass
    self.contact_constraint = None
    self.grasped_object = None
    self.activated = False
    self._set_fingers(0.0)

  def detect_contact(self):
    """Detect a pick candidate, or contact made by the held object."""
    if self.grasped_object is not None:
      points = p.getContactPoints(bodyA=self.grasped_object)
      return any(point[2] not in (self.body, self.robot) for point in points)
    return self._candidate() is not None

  def check_grasp(self):
    return self.grasped_object is not None


class Suction(Gripper):
  """Simulate simple suction dynamics."""

  def __init__(self, assets_root, robot, ee, obj_ids):
    """Creates suction and 'attaches' it to the robot.

    Has special cases when dealing with rigid vs deformables. For rigid,
    only need to check contact_constraint for any constraint. For soft
    bodies (i.e., cloth or bags), use cloth_threshold to check distances
    from gripper body (self.body) to any vertex in the cloth mesh. We
    need correct code logic to handle gripping potentially a rigid or a
    deformable (and similarly for releasing).

    To be clear on terminology: 'deformable' here should be interpreted
    as a PyBullet 'softBody', which includes cloths and bags. There's
    also cables, but those are formed by connecting rigid body beads, so
    they can use standard 'rigid body' grasping code.

    To get the suction gripper pose, use p.getLinkState(self.body, 0),
    and not p.getBasePositionAndOrientation(self.body) as the latter is
    about z=0.03m higher and empirically seems worse.

    Args:
      assets_root: str for root directory with assets.
      robot: int representing PyBullet ID of robot.
      ee: int representing PyBullet ID of end effector link.
      obj_ids: list of PyBullet IDs of all suctionable objects in the env.
    """
    super().__init__(assets_root)

    # Load suction gripper base model (visual only).
    pose = ((0.487, 0.109, 0.438), p.getQuaternionFromEuler((np.pi, 0, 0)))
    base = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_BASE_URDF), pose[0], pose[1])
    p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=base,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, 0.01))

    # Load suction tip model (visual and collision) with compliance.
    # urdf = 'assets/ur5/suction/suction-head.urdf'
    pose = ((0.487, 0.109, 0.347), p.getQuaternionFromEuler((np.pi, 0, 0)))
    self.body = pybullet_utils.load_urdf(
        p, os.path.join(self.assets_root, SUCTION_HEAD_URDF), pose[0], pose[1])
    constraint_id = p.createConstraint(
        parentBodyUniqueId=robot,
        parentLinkIndex=ee,
        childBodyUniqueId=self.body,
        childLinkIndex=-1,
        jointType=p.JOINT_FIXED,
        jointAxis=(0, 0, 0),
        parentFramePosition=(0, 0, 0),
        childFramePosition=(0, 0, -0.08))
    p.changeConstraint(constraint_id, maxForce=50)

    # Reference to object IDs in environment for simulating suction.
    self.obj_ids = obj_ids

    # Indicates whether gripper is gripping anything (rigid or def).
    self.activated = False

    # For gripping and releasing rigid objects.
    self.contact_constraint = None

    # Defaults for deformable parameters, and can override in tasks.
    self.def_ignore = 0.035  # TODO(daniel) check if this is needed
    self.def_threshold = 0.030
    self.def_nb_anchors = 1

    # Track which deformable is being gripped (if any), and anchors.
    self.def_grip_item = None
    self.def_grip_anchors = []

    # Determines release when gripped deformable touches a rigid/def.
    # TODO(daniel) should check if the code uses this -- not sure?
    self.def_min_vetex = None
    self.def_min_distance = None

    # Determines release when a gripped rigid touches defs (e.g. cloth-cover).
    self.init_grip_distance = None
    self.init_grip_item = None

  def activate(self):
    """Simulate suction using a rigid fixed constraint to contacted object."""
    # TODO(andyzeng): check deformables logic.
    # del def_ids

    if not self.activated:
      points = p.getContactPoints(bodyA=self.body, linkIndexA=0)
      # print(points)
      if points:

        # Handle contact between suction with a rigid object.
        for point in points:
          obj_id, contact_link = point[2], point[4]
        if obj_id in self.obj_ids['rigid']:
          body_pose = p.getLinkState(self.body, 0)
          obj_pose = p.getBasePositionAndOrientation(obj_id)
          world_to_body = p.invertTransform(body_pose[0], body_pose[1])
          obj_to_body = p.multiplyTransforms(world_to_body[0],
                                             world_to_body[1],
                                             obj_pose[0], obj_pose[1])
          self.contact_constraint = p.createConstraint(
              parentBodyUniqueId=self.body,
              parentLinkIndex=0,
              childBodyUniqueId=obj_id,
              childLinkIndex=contact_link,
              jointType=p.JOINT_FIXED,
              jointAxis=(0, 0, 0),
              parentFramePosition=obj_to_body[0],
              parentFrameOrientation=obj_to_body[1],
              childFramePosition=(0, 0, 0),
              childFrameOrientation=(0, 0, 0))

        self.activated = True

  def release(self):
    """Release gripper object, only applied if gripper is 'activated'.

    If suction off, detect contact between gripper and objects.
    If suction on, detect contact between picked object and other objects.

    To handle deformables, simply remove constraints (i.e., anchors).
    Also reset any relevant variables, e.g., if releasing a rigid, we
    should reset init_grip values back to None, which will be re-assigned
    in any subsequent grasps.
    """
    if self.activated:
      self.activated = False

      # Release gripped rigid object (if any).
      if self.contact_constraint is not None:
        try:
          p.removeConstraint(self.contact_constraint)
          self.contact_constraint = None
        except:  # pylint: disable=bare-except
          pass
        self.init_grip_distance = None
        self.init_grip_item = None

      # Release gripped deformable object (if any).
      if self.def_grip_anchors:
        for anchor_id in self.def_grip_anchors:
          p.removeConstraint(anchor_id)
        self.def_grip_anchors = []
        self.def_grip_item = None
        self.def_min_vetex = None
        self.def_min_distance = None

  def detect_contact(self):
    """Detects a contact with a rigid object."""
    body, link = self.body, 0
    if self.activated and self.contact_constraint is not None:
      try:
        info = p.getConstraintInfo(self.contact_constraint)
        body, link = info[2], info[3]
      except:  # pylint: disable=bare-except
        self.contact_constraint = None
        pass

    # Get all contact points between the suction and a rigid body.
    points = p.getContactPoints(bodyA=body, linkIndexA=link)
    # print(points)
    # exit()
    if self.activated:
      points = [point for point in points if point[2] != self.body]

    # # We know if len(points) > 0, contact is made with SOME rigid item.
    if points:
      return True

    return False

  def check_grasp(self):
    """Check a grasp (object in contact?) for picking success."""

    suctioned_object = None
    if self.contact_constraint is not None:
      suctioned_object = p.getConstraintInfo(self.contact_constraint)[2]
    return suctioned_object is not None
