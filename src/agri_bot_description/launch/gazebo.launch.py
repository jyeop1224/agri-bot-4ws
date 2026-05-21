import os
from launch import LaunchDescription
from launch.actions import ExecuteProcess, TimerAction, SetEnvironmentVariable, DeclareLaunchArgument
from launch.substitutions import Command, LaunchConfiguration
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():       
    pkg = get_package_share_directory('agri_bot_description')
    urdf_file = os.path.join(pkg, 'urdf', 'agri_bot.urdf.xacro')

    # world 파라미터 선언 (기본값: 기존 3행 환경)
    declare_world = DeclareLaunchArgument(
        'world',
        default_value='greenhouse.world',
        description='World file name (in agri_bot_description/worlds/)'
    )

    # spawn_y 파라미터 선언 (기본값: 통로1 중앙)
    declare_spawn_y = DeclareLaunchArgument(
        'spawn_y',
        default_value='1.0',
        description='Robot spawn y position (통로1 중앙)'
    )

    world_file = [
        os.path.join(pkg, 'worlds', ''),
        LaunchConfiguration('world')
    ]

    use_sim_time = {'use_sim_time': True}

    gazebo_plugin_path = SetEnvironmentVariable(
        name='GAZEBO_PLUGIN_PATH',
        value='/opt/ros/humble/lib:/opt/ros/humble/lib/x86_64-linux-gnu'
    )

    gzserver = ExecuteProcess(
        cmd=['gzserver', '--verbose', world_file,
             '-s', 'libgazebo_ros_init.so',
             '-s', 'libgazebo_ros_factory.so'],
        output='screen'
    )

    gzclient = ExecuteProcess(
        cmd=['gzclient', '--verbose'],
        output='screen'
    )

    robot_state_publisher = TimerAction(
        period=3.0,
        actions=[
            Node(
                package='robot_state_publisher',
                executable='robot_state_publisher',
                output='screen',
                parameters=[{
                    'robot_description': Command(['xacro ', urdf_file]),
                    **use_sim_time,
                }]
            )
        ]
    )

    spawn_robot = TimerAction(
        period=10.0,
        actions=[
            Node(
                package='gazebo_ros',
                executable='spawn_entity.py',
                arguments=[
                    '-topic', 'robot_description',
                    '-entity', 'agri_bot',
                    '-x', '3.0',
                    '-y', LaunchConfiguration('spawn_y'),
                    '-z', '0.15',
                    '-R', '0.0', '-P', '0.0', '-Y', '0.0',
                ],
                output='screen',
            )
        ]
    )

    load_jsb = TimerAction(
        period=16.0,
        actions=[Node(
            package='controller_manager', executable='spawner',
            arguments=['joint_state_broadcaster',
                       '--controller-manager', '/controller_manager'],
            output='screen',
        )]
    )

    load_steer = TimerAction(
        period=18.0,
        actions=[Node(
            package='controller_manager', executable='spawner',
            arguments=['steering_controller',
                       '--controller-manager', '/controller_manager'],
            output='screen',
        )]
    )

    load_wheel = TimerAction(
        period=20.0,
        actions=[Node(
            package='controller_manager', executable='spawner',
            arguments=['wheel_controller',
                       '--controller-manager', '/controller_manager'],
            output='screen',
        )]
    )

    return LaunchDescription([
        declare_world,
        declare_spawn_y,
        gazebo_plugin_path,
        gzserver,
        gzclient,
        robot_state_publisher,
        spawn_robot,
        load_jsb,
        load_steer,
        load_wheel,
    ])