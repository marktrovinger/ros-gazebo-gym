"""Template file used that can be used to create a new Robot environment. It contains a
Python class that specifies the robot to use on the task. It provides the complete
integration between the Gazebo simulation of the robot and the gymnasium library, so
obtaining **SENSOR** information from the robot or sending **ACTIONS** to it are
transparent to the gymnasium library and you, the developer. For more information, see
the `ros_gazebo_gym <https://rickstaa.dev/ros-gazebo-gym>`_ documentation.

Source code
-----------

.. literalinclude:: ../../../../../templates/template_my_robot_env.py
   :language: python
   :linenos:
   :lines: 16-
"""
import copy
from datetime import datetime
from itertools import compress

import actionlib
import numpy as np
import rospy
import tf2_ros
from control_msgs.msg import GripperCommandAction, GripperCommandGoal
from geometry_msgs.msg import (
    Pose,
    Quaternion,
    Vector3,
    TransformStamped,
    Transform,
    Point,
)
from tf.transformations import quaternion_inverse, quaternion_multiply
from ros_gazebo_gym.common.helpers import (
    action_server_exists,
    flatten_list,
    get_orientation_euler,
    is_sublist,
    list_2_human_text,
    lower_first_char,
    normalize_quaternion,
    suppress_stderr,
)
from ros_gazebo_gym.core.helpers import get_log_path, ros_exit_gracefully
from ros_gazebo_gym.core.lazy_importer import LazyImporter
from ros_gazebo_gym.core.ros_launcher import ROSLauncher
from ros_gazebo_gym.exceptions import EePoseLookupError, EeRpyLookupError
from ros_gazebo_gym.robot_envs.helpers import (
    remove_gripper_commands_from_joint_commands_msg,
)
from ros_gazebo_gym.robot_gazebo_goal_env import RobotGazeboGoalEnv
from rospy.exceptions import ROSException, ROSInterruptException
from sensor_msgs.msg import JointState
from std_msgs.msg import Float32, Float64MultiArray, Header
from urdf_parser_py.urdf import URDF
from tf2_geometry_msgs import PoseStamped
from tf2_ros import (
    StaticTransformBroadcaster,
)

class KinovaEnv(RobotGazeboGoalEnv):
    """Superclass for all Robot environments."""

    def __init__(self):
        """Initializes a new Robot environment."""
        # Setup internal robot environment variables (controllers, namespace etc.).
        # NOTE: If the controllers_list is not set all the currently running controllers
        # will be reset by the ControllersList class.
        # TODO: Change the controller list and robot namespace.
        self.controllers_list = [
            "left_gen3_joint_trajectory_controller",
            "right_gen3_joint_trajectory_controller",
        ]
        self.robot_name_space = "my_gen3"
        reset_controls_bool = True or False

        # Initialize the gazebo environment.
        super(KinovaEnv, self).__init__(
            controllers_list=self.controllers_list,
            robot_name_space=self.robot_name_space,
            reset_controls=reset_controls_bool,
        )

    ################################################
    # Overload Gazebo env virtual methods ##########
    ################################################
    # NOTE: Methods that need to be implemented as they are called by the robot and
    # Gazebo environments.
    def _check_all_systems_ready(self):
        """Checks that all the sensors, publishers and other simulation systems are
        operational.
        """
        # TODO: Implement the logic that checks if all sensors and actuators are ready.
        
        return True

    ################################################
    # Robot environment internal methods ###########
    ################################################
    # NOTE: Here you can add additional helper methods that are used in the robot env.

    ################################################
    # Robot env main methods #######################
    ################################################
    # NOTE: Contains methods that the task environment will need.
