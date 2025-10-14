from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, RegisterEventHandler, ExecuteProcess
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch.event_handlers import OnProcessExit
from launch.substitutions import EnvironmentVariable
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():
    package_name = 'orbbec_hand_detection'
    pkg_share = get_package_share_directory(package_name)

    rviz_config_path = os.path.join(
    get_package_share_directory('orbbec_hand_detection'), 
    'rviz', 
    'pointer_depth_config.rviz') 

    # Calibration file path
    config_dir = os.path.join(pkg_share, 'config')
    calibration_file = os.path.join(config_dir, 'calibration.yml')

    # Ensure config directory exists
    os.makedirs(config_dir, exist_ok=True)

    # Delete old calibration file if it exists
    if os.path.exists(calibration_file):
        print(f"[INFO] Removing old calibration file: {calibration_file}")
        os.remove(calibration_file)

    # --- Include Orbbec camera launch ---
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('orbbec_camera'),
                'launch',
                'gemini2.launch.py'
            )
        ),
        launch_arguments={
            'color_width': '1280',
            'color_height': '720',
            'color_fps': '10',
            'depth_width': '1280',
            'depth_height': '800',
            'depth_fps': '10',
            'depth_registration': 'True'
        }.items()
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

    # These nodes starts after calibration is done
    pointer_tip_depth_plane_node = Node(
        package=package_name,
        executable='pointer_tip_depth_plane',
        name='pointer_tip_depth_plane',
        parameters=[
            {'calib_yaml_path': calibration_file},
            {'nats_url': EnvironmentVariable('NATS_URL', default_value='nats://localhost:4222')},
            {'screen_width': EnvironmentVariable('SCREEN_WIDTH', default_value='1920')},
            {'screen_height': EnvironmentVariable('SCREEN_HEIGHT', default_value='1080')},
            {'depth_threshold': EnvironmentVariable('DEPTH_THR', default_value='0.01')},
            {'min_detection_confidence': EnvironmentVariable('MIN_DET_CONF', default_value='0.5')},
            {'min_tracking_confidence': EnvironmentVariable('MIN_TRACK_CONF', default_value='0.5')},
        ],
        output='screen'
    )

    # Rviz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
        )  
    

    # Event handler: when calibration node exits, stop ArUco display node and start other nodes
    stop_calibration_and_start_detection = RegisterEventHandler(
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
                ),
                pointer_tip_depth_plane_node,
            ],
        )
    )

    return LaunchDescription([
        rviz_node,
        camera_launch,
        aruco_display_node,
        screen_calibration_node,
        stop_calibration_and_start_detection,
       
    ])

