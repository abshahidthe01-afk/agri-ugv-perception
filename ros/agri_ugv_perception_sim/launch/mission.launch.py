"""
agri-ugv-sim's field mission, with the extras of make_field_extras added to the world.

    ros2 launch agri_ugv_perception_sim mission.launch.py plots:=198 cameras:=top

Starts agri_ugv_navigation's mission.launch.py with the same arguments (plots, cameras,
camera_depth, scanners, gui, ...: see there) and adds the model that make_field_extras wrote
(extras: its folder, default ~/agri_data/models/field_extras) to the running world, the way
agri_ugv_gazebo's sim.launch.py adds the robot.
"""

import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch_ros.actions import Node


def generate_launch_description():
    mission = os.path.join(get_package_share_directory('agri_ugv_navigation'), 'launch',
                           'mission.launch.py')
    extras = os.path.join(os.path.expanduser('~'), 'agri_data', 'models', 'field_extras')
    return LaunchDescription([
        DeclareLaunchArgument('extras', default_value=extras,
                              description='folder of the model written by make_field_extras'),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(mission)),
        Node(package='ros_gz_sim', executable='create', output='screen',
             arguments=['-file', PathJoinSubstitution([LaunchConfiguration('extras'),
                                                       'model.sdf']),
                        '-name', 'field_extras']),
    ])
