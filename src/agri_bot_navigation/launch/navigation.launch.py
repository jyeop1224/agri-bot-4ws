import os
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    nav_pkg = get_package_share_directory('agri_bot_navigation')
    nav2_pkg = get_package_share_directory('nav2_bringup')

    map_file = os.path.join(nav_pkg, 'maps', 'greenhouse.yaml')
    param_file = os.path.join(nav_pkg, 'param', 'agri_bot.yaml')

    nav2 = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(nav2_pkg, 'launch', 'bringup_launch.py')
        ),
        launch_arguments={
            'map': map_file,
            'params_file': param_file,
            'use_sim_time': 'true',
        }.items()
    )

    rviz_config = os.path.join(
        nav2_pkg, 'rviz', 'nav2_default_view.rviz'
    )

    rviz = Node(
        package='rviz2',
        executable='rviz2',
        arguments=['-d', rviz_config],
        output='screen',
    )

    return LaunchDescription([nav2, rviz])
