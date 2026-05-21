from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    use_sim_time = {'use_sim_time': True}

    return LaunchDescription([
        Node(
            package='agri_bot_4ws',
            executable='four_ws_controller',
            output='screen',
            parameters=[use_sim_time]
        ),
        Node(
            package='agri_bot_4ws',
            executable='odometry_publisher',
            output='screen',
            parameters=[use_sim_time]
        ),
    ])
