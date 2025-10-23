from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'orbbec_hand_detection'
    pkg_share = get_package_share_directory(package_name)

    # Calibration file path
    config_dir = os.path.join(pkg_share, 'config')
    calibration_file = os.path.join(config_dir, 'calibration.yml')

    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Delete old calibration file if it exists
    if os.path.exists(calibration_file):
        print(f"[INFO] Removing old calibration file: {calibration_file}")
        os.remove(calibration_file)

    # Include the orbbec camera launch
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('orbbec_camera'),
                'launch',
                'gemini2.launch.py'
            )
        )
    )

    # Nodes
    aruco_display_node = Node(
        package=package_name,
        executable='aruco_display_node',
        name='aruco_display_node',
        output='screen'
    )

    screen_calibration_node = Node(
        package=package_name,
        executable='screen_calibration_node',
        name='screen_calibration_node',
        parameters=[
            {'rgb_topic': '/camera/color/image_raw'},
            {'depth_topic': '/camera/depth/image_raw'},
            {'depth_info_topic': '/camera/depth/camera_info'},
            {'output_yaml': calibration_file}
        ],
        output='screen'
    )

    # Event handler: when calibration node exits, stop ArUco display node
    stop_aruco_on_calibration_done = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=screen_calibration_node,
            on_exit=[
                ExecuteProcess(
                    cmd=['ros2', 'node', 'kill', '/aruco_display_node'],
                    shell=True
                ),
                ExecuteProcess(
                    cmd=['echo', '[INFO] Calibration finished. ArUco display node stopped.'],
                    shell=True
                )
            ],
        )
    )

    return LaunchDescription([
        camera_launch,
        aruco_display_node,
        screen_calibration_node,
        stop_aruco_on_calibration_done
    ])
