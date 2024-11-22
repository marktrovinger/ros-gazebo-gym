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

# Like most of the code, this was taken from panda_env.py,
# need to make sure that variables are changed to Kinova specific
# TODO: Kinova topics
CONNECTION_TIMEOUT = 5  # Timeout for connecting to services or topics.
GAZEBO_SIM_CONNECTION_TIMEOUT = 60  # Timeout for waiting for gazebo to be launched.
MOVEIT_SET_EE_POSE_TOPIC = "panda_moveit_planner_server/panda_arm/set_ee_pose"
MOVEIT_GET_EE_POSE_JOINT_CONFIG_TOPIC = (
    "panda_moveit_planner_server/panda_arm/get_ee_pose_joint_config"
)
LOCK_UNLOCK_TOPIC = "lock_unlock_panda_joints"
GET_CONTROLLED_JOINTS_TOPIC = "panda_control_server/get_controlled_joints"
SET_JOINT_COMMANDS_TOPIC = "panda_control_server/set_joint_commands"
SET_GRIPPER_WIDTH_TOPIC = "panda_control_server/panda_hand/set_gripper_width"
SET_JOINT_TRAJECTORY_TOPIC = "panda_control_server/panda_arm/follow_joint_trajectory"
FRANKA_GRIPPER_COMMAND_TOPIC = "franka_gripper/gripper_action"
JOINT_STATES_TOPIC = "joint_states"
KINOVA_STATES_TOPIC = "my_gen3/franka_states"
GET_PANDA_EE_FRAME_TRANSFORM_TIMEOUT = (
    1  # Timeout for retrieving the panda EE frame transform.  # noqa: E501
)

# Other script variables.
AVAILABLE_CONTROL_TYPES = [
    "trajectory",
    "position",
    "effort",
    "end_effector",
]
PANDA_JOINTS_FALLBACK = {
    "arm": [
        "panda_joint1",
        "panda_joint2",
        "panda_joint3",
        "panda_joint4",
        "panda_joint5",
        "panda_joint6",
        "panda_joint7",
    ],
    "hand": ["panda_finger_joint1", "panda_finger_joint2"],
}  # NOTE: Used when the joints can not be determined.
ARM_POSITION_CONTROLLER = "panda_arm_joint_position_controller"
ARM_EFFORT_CONTROLLER = "panda_arm_joint_effort_controller"
GRASP_FORCE = 10  # Default panda gripper force. Panda force information: {Continuous force: 70N, max_force: 140 N}.  # noqa: E501
ARM_CONTROL_WAIT_TIMEOUT = 5  # Default arm control wait timeout [s].
ARM_JOINT_POSITION_WAIT_THRESHOLD = 0.07  # Threshold used for determining whether a joint position is reached (i.e. 0.01 rad per joint).  # noqa: E501
ARM_JOINT_EFFORT_WAIT_THRESHOLD = 7  # Threshold used for determining whether a joint position is reached (i.e. 1 N per joint).  # noqa: E501
ARM_JOINT_VELOCITY_WAIT_THRESHOLD = 0.07  # Threshold used for determining whether the joint velocity is zero (i.e. 1rad/s per joint).  # noqa: E501


class KinovaEnv(RobotGazeboGoalEnv):
    """Superclass for all Robot environments."""

    def __init__(self,
        robot_name_space="",
        robot_EE_link="robotiq_2f_85",
        ee_frame_offset=None,
        load_gripper=True,
        lock_gripper=False,
        grasping=False,
        control_type="effort",
        reset_robot_pose=True,
        workspace_path=None,
        log_reset=True,
        visualize=None,         
    ):
        """Initializes a new Kinova Robot environment."""
        # Setup internal robot environment variables (controllers, namespace etc.).
        # NOTE: If the controllers_list is not set all the currently running controllers
        # will be reset by the ControllersList class.
        # TODO: Change the controller list and robot namespace.
        rospy.logwarn("Initializing KinovaEnv robot environment...")

        # Import lazy
        self.kinova_gazebo = LazyImporter("kortex_gazebo")

        self.robot_name_space = robot_name_space
        self.reset_controls = True
        self.robot_EE_link = robot_EE_link
        self.load_gripper = load_gripper
        self.lock_gripper = lock_gripper
        self._ee_frame_offset_dict = ee_frame_offset
        self._ros_shutdown_requested = False
        self._connection_timeout = CONNECTION_TIMEOUT
        self._joint_traj_action_server_default_step_size = 1
        self._grasping = grasping
        self._direct_control = (
            True if not hasattr(self, "_direct_control") else self._direct_control
        )
        self._log_step_debug_info = (
            False
            if not hasattr(self, "_log_step_debug_info")
            else self._log_step_debug_info
        )
        self._joint_lock_client_connected = False
        self._get_controlled_joints_client_connected = False
        self._moveit_set_ee_pose_client_connected = False
        self._moveit_get_ee_pose_joint_config_client_connected = False
        self._arm_joint_traj_control_client_connected = False
        self._set_joint_commands_client_connected = False
        self._set_gripper_width_client_connected = False
        self._fetched_joints = False
        self._ros_is_shutdown = False
        self._last_gripper_goal = None
        self.__robot_control_type = control_type.lower()
        self.__joints = {}
        self.__in_collision = False
        self.__locked_joints = []
    
        self.controllers_list = [
            "left_gen3_joint_trajectory_controller",
            "right_gen3_joint_trajectory_controller",
        ]
        self.robot_name_space = "my_gen3"
        reset_controls_bool = True or False

        # Thrown control warnings.
        if self._direct_control and self.robot_control_type in [
            "trajectory",
            "end_effector",
        ]:
            rospy.logwarn(
                "Direct control variable 'direct_control' was ignored as it "
                f"is not implemented for '{self.robot_control_type}' control."
            )
        if self.robot_control_type == "position":
            rospy.logwarn(
                "Position control is experimental and not yet fully implemented. "
                "See https://github.com/rickstaa/panda-gazebo/issues/12."
            )

        simulation_check_timeout_time = rospy.get_rostime() + rospy.Duration(
            GAZEBO_SIM_CONNECTION_TIMEOUT
        )
        while (
            not rospy.is_shutdown()
            and rospy.get_rostime() < simulation_check_timeout_time
        ):
            if any(
                [
                    "/gazebo" in topic
                    for topic in flatten_list(rospy.get_published_topics())
                ]
            ):
                break
            else:
                rospy.logwarn_once("Waiting for the Gazebo simulation to be started...")
        else:
            if not rospy.is_shutdown():
                err_msg = (
                    f"Shutting down '{rospy.get_name()}' since the Kinova Gazebo "
                    "simulation was not started within the set timeout period of "
                    f"{GAZEBO_SIM_CONNECTION_TIMEOUT} seconds."
                )
                ros_exit_gracefully(shutdown_msg=err_msg, exit_code=1)

        # Validate requested control type.
        if self.robot_control_type not in AVAILABLE_CONTROL_TYPES:
            err_msg = (
                f"Shutting down '{rospy.get_name()}' because control type "
                f"'{control_type}' that was specified is invalid. Please use one of "
                "the following robot control types and try again: "
            )
            for ctrl_type in AVAILABLE_CONTROL_TYPES:
                err_msg += f"\n - {ctrl_type}"
            ros_exit_gracefully(shutdown_msg=err_msg, exit_code=1)
        else:
            rospy.logwarn(f"Kinova robot is controlled using '{control_type}' control.")

        # Launch the ROS launch that spawns the robot into the world.
        control_type_group = (
            "trajectory"
            if self.robot_control_type == "end_effector"
            else self.robot_control_type
        )  # NOTE: Ee control uses the trajectory controllers.
        launch_log_file = str(
            get_log_path().joinpath(
                "put_robot_in_world_launch_{}.log".format(
                    datetime.now().strftime("%d_%m_%Y_%H_%M_%S"),
                )
            )
            if (
                hasattr(self, "_roslaunch_log_to_console")
                and not self._roslaunch_log_to_console
            )
            else None
        )
        show_rviz = (
            visualize
            if visualize is not None
            else (self._load_rviz if hasattr(self, "_load_rviz") else True)
        )
        ROSLauncher.launch(
            package_name="kortex_gazebo",
            launch_file_name="spawn_kortex_robot.launch",
            workspace_path=workspace_path,
            log_file=launch_log_file,
            critical=True,
            rviz=show_rviz,
            load_gripper=self.load_gripper,
            disable_franka_gazebo_logs=True,
            rviz_file=self._rviz_file if hasattr(self, "_rviz_file") else "",
            end_effector=self.robot_EE_link,
            max_velocity_scaling_factor=(
                self._max_velocity_scaling_factor
                if hasattr(self, "_max_velocity_scaling_factor")
                else ""
            ),
            max_acceleration_scaling_factor=(
                self._max_acceleration_scaling_factor
                if hasattr(self, "_max_acceleration_scaling_factor")
                else ""
            ),
            control_type=control_type_group,
        )
        # Add ros shutdown hook.
        rospy.on_shutdown(self._ros_shutdown_hook)

        # Initialize the gazebo environment.
        super(KinovaEnv, self).__init__(
            robot_name_space=self.robot_name_space,
            reset_controls=self.reset_controls,
            reset_robot_pose=reset_robot_pose,
            reset_world_or_sim="WORLD",
            log_reset=log_reset,
            pause_simulation=(
                self._pause_after_step if hasattr(self, "_pause_after_step") else False
            ),
            publish_rviz_training_info_overlay=(
                self._load_rviz if hasattr(self, "_load_rviz") else True
            ),
        )

        ########################################
        # Initialize sensor topics #############
        ########################################

        # Create publishers.
        rospy.logdebug("Creating publishers.")
        self._in_collision_pub = rospy.Publisher(
            "/ros_gazebo_gym/in_collision", Float32, queue_size=1, latch=True
        )

        # Create joint state and franka state subscriber.
        rospy.logdebug("Connecting to sensors.")
        rospy.Subscriber(
            f"{self.robot_name_space}/{JOINT_STATES_TOPIC}",
            JointState,
            self._joint_states_cb,
            queue_size=1,
        )
        rospy.Subscriber(
            KINOVA_STATES_TOPIC,
            self.franka_msgs.msg.FrankaState,
            self._franka_states_cb,
            queue_size=1,
        )

        # Create transform listener.
        rospy.logdebug("Creating tf2 buffer.")
        self.tf_buffer = tf2_ros.Buffer()
        self._tf_listener = tf2_ros.TransformListener(self.tf_buffer)
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
