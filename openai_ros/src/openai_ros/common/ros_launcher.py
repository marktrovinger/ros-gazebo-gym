"""Launches all the ROS nodes that are needed for a given openai_sim gym environment.
"""
import atexit
import subprocess
from pathlib import Path

import catkin
import rospy
from openai_ros.common.helpers import get_global_pkg_path, package_installer


class ROSLauncher(object):
    """Class used to launch ROS launch files.

    Attributes:
        successful (bool): Whether the launch file was successfully launched. This only
            specifies if the launchfile was successfully executed not if it encountered
            errors.
    """

    launched = {}  # Stores all processes that were launched

    @classmethod
    def launch(cls, package_name, launch_file_name, workspace_path=None):
        """Launch a given launchfile while also installing the launchfile package and or
        dependencies. This is done by using the openai_ros dependency index.

        Args:
            package_name (str): The package that contains the launchfile.
            launch_file_name (str): The launchfile name.
            workspace_path (str, optional): The path of the catkin workspace. Defaults
                to ``None`` meaning the path will be determined.

        Raises:
            Exception: When something went wrong when launching the launchfile.
        """
        # Install launch file dependencies if they are not present
        try:
            package_installed = package_installer(
                package_name, workspace_path=workspace_path
            )
        except Exception:
            rospy.logwarn(
                f"Something went wrong while trying to install the '{package_name}' "
                "ROS package and its dependencies."
            )
            package_installed = False

        # Retrieve workspace path
        workspace_path = (
            workspace_path
            if workspace_path
            else catkin.workspace.get_workspaces()[0].replace("/devel", "")
        )

        # Launch launch file if package was found
        if package_installed:
            rospy.loginfo(
                f"Starting '{launch_file_name}' launch file from package "
                f"'{package_name}."
            )
            pkg_name = get_global_pkg_path(package_name)
            if pkg_name:
                launch_dir = Path(get_global_pkg_path(package_name)).joinpath("launch")
                path_launch_file_name = Path(launch_dir).joinpath(launch_file_name)
                rospy.logdebug(f"Launch file path: {path_launch_file_name}")
            source_command = ". {} {};".format(
                workspace_path, Path("/devel/setup.sh").resolve()
            )
            roslaunch_command = "roslaunch {} {}".format(package_name, launch_file_name)
            command = source_command + roslaunch_command
            rospy.logwarn("Launching command: " + str(command))

            # Launch the launchfile using a subprocess.
            # NOTE: I also tried using the roslaunch python api but I could not find a
            # way to first source the catkin workspace.
            p = subprocess.Popen(command, shell=True)
            state = p.poll()
            if state is None:
                rospy.logdebug("Launch file successfully launched.")
                cls.launched[launch_file_name] = p
                atexit.register(p.kill)  # Make sure process dies when parent dies
            elif state < 0:
                rospy.logerror(
                    "Something went wrong while trying to launch the "
                    f"{path_launch_file_name} launch file."
                )
            elif state > 0:
                rospy.logerror(
                    "Something went wrong while trying to launch the "
                    f"{path_launch_file_name} launch file."
                )
        else:
            raise Exception(f"Package '{package_name}' could not be found.")
