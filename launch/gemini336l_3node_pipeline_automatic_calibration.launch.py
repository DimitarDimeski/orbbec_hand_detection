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

    # Include Orbbec camera launch
    camera_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('orbbec_camera'),
                'launch',
                'gemini_330_series.launch.py'
            )
        ),
        launch_arguments={
            'color_width': EnvironmentVariable('COLOR_WIDTH', default_value='1280'),
            'color_height': EnvironmentVariable('COLOR_HEIGHT', default_value='720'),
            'color_fps': EnvironmentVariable('COLOR_FPS', default_value='10'),
            'depth_width': EnvironmentVariable('DEPTH_WIDTH', default_value='1280'),
            'depth_height': EnvironmentVariable('DEPTH_HEIGHT', default_value='800'),
            'device_preset': EnvironmentVariable('DEVICE_PRESET', default_value='Default'),
            'depth_fps': EnvironmentVariable('DEPTH_FPS', default_value='10'),
            'depth_registration': 'True',
            'enable_decimation_filter': EnvironmentVariable('ENABLE_DECIMATION_FILTER', default_value='False'),
            'enable_spatial_filter': EnvironmentVariable('ENABLE_SPATIAL_FILTER', default_value='False'),
            'enable_temporal_filter': EnvironmentVariable('ENABLE_TEMPORAL_FILTER', default_value='False'),
            'enable_hole_filling_filter': EnvironmentVariable('ENABLE_HOLE_FILLING_FILTER', default_value='False'),
            'enable_color_auto_exposure': EnvironmentVariable('ENABLE_COLOR_AUTO_EXPOSURE', default_value='True'),
            'color_exposure': EnvironmentVariable('COLOR_EXPOSURE', default_value='-1'),
            'color_gain': EnvironmentVariable('COLOR_GAIN', default_value='-1'),
            'enable_color_auto_white_balance': EnvironmentVariable('ENABLE_COLOR_AUTO_WHITE_BALANCE', default_value='True'),
            'color_white_balance': EnvironmentVariable('COLOR_WHITE_BALANCE', default_value='-1'),
            'color_brightness': EnvironmentVariable('COLOR_BRIGHTNESS', default_value='-1'),
            'enable_color_decimation_filter': EnvironmentVariable('ENABLE_COLOR_DECIMATION_FILTER', default_value='False'),
            'color_decimation_filter_scale': EnvironmentVariable('COLOR_DECIMATION_FILTER_SCALE', default_value='-1'),
        }.items()
    )

    # Nodes
    aruco_display_node = Node(
        package=package_name,
        executable='aruco_display_node',
        name='aruco_display_node',
        parameters=[
            {'output_yaml': calibration_file},
            {'marker_size' : EnvironmentVariable('MARKER_SIZE', default_value='200')},
            {'top_offset' : EnvironmentVariable('TOP_OFFSET', default_value='0')},
            {'bottom_offset' : EnvironmentVariable('BOTTOM_OFFSET', default_value='0')},
            {'left_offset' : EnvironmentVariable('LEFT_OFFSET', default_value='0')},
            {'right_offset' : EnvironmentVariable('RIGHT_OFFSET', default_value='0')},
        ],
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
            {'output_yaml': calibration_file},
            {'rotate_image': EnvironmentVariable('ROTATE_IMAGE', default_value='False')},
        ],
        output='screen'
    )

    # These nodes start after calibration is done
    
    # Node 1: MediaPipe Detection Node
    mediapipe_detection_node = Node(
        package=package_name,
        executable='mediapipe_detection_node',
        name='mediapipe_detection_node',
        parameters=[
            {'rgb_topic': EnvironmentVariable('RGB_TOPIC', default_value='/camera/color/image_raw')},
            {'detection_topic': EnvironmentVariable('DETECTION_TOPIC', default_value='/hand_detections')},
            {'min_detection_confidence': EnvironmentVariable('MIN_DET_CONF', default_value='0.5')},
            {'min_tracking_confidence': EnvironmentVariable('MIN_TRACK_CONF', default_value='0.5')},
            {'static_image_mode': EnvironmentVariable('STATIC_IMAGE_MODE', default_value='False')},
            {'max_num_hands': EnvironmentVariable('MAX_NUM_HANDS', default_value='10')},
            {'model_complexity': EnvironmentVariable('MODEL_COMPLEXITY', default_value='1')},
            {'rotate_image': EnvironmentVariable('ROTATE_IMAGE', default_value='False')},
            {'use_depth': EnvironmentVariable('USE_DEPTH', default_value='False')},
            {'use_grayscale': EnvironmentVariable('USE_GRAYSCALE', default_value='False')},
            {'adjust_contrast_brightness': EnvironmentVariable('ADJUST_CONTRAST_BRIGHTNESS', default_value='False')},
            {'contrast': EnvironmentVariable('CONTRAST', default_value='1.5')},
            {'brightness': EnvironmentVariable('BRIGHTNESS', default_value='20')},
            {'debug': EnvironmentVariable('DEBUG', default_value='False')},
        ],
        output='screen'
    )

    # Node 2: Touch Detection Node
    touch_detection_node = Node(
        package=package_name,
        executable='touch_detection_node',
        name='touch_detection_node',
        parameters=[
            {'depth_topic': EnvironmentVariable('DEPTH_TOPIC', default_value='/camera/depth/image_raw')},
            {'detection_topic': EnvironmentVariable('DETECTION_TOPIC', default_value='/hand_detections')},
            {'touch_detections_topic': EnvironmentVariable('TOUCH_DETECTIONS_TOPIC', default_value='/touch_detections')},
            {'calib_yaml_path': calibration_file},
            {'screen_width': EnvironmentVariable('SCREEN_WIDTH', default_value='1920')},
            {'screen_height': EnvironmentVariable('SCREEN_HEIGHT', default_value='1080')},
            {'depth_threshold': EnvironmentVariable('DEPTH_THR', default_value='0.01')},
            {'depth_threshold_lower': EnvironmentVariable('DEPTH_THR_LOWER', default_value='0.5')},
            {'depth_threshold_upper': EnvironmentVariable('DEPTH_THR_UPPER', default_value='0.5')},
            {'top_offset': EnvironmentVariable('TOP_OFFSET', default_value='0')},
            {'bottom_offset': EnvironmentVariable('BOTTOM_OFFSET', default_value='0')},
            {'left_offset': EnvironmentVariable('LEFT_OFFSET', default_value='0')},
            {'right_offset': EnvironmentVariable('RIGHT_OFFSET', default_value='0')},
            {'rotate_image': EnvironmentVariable('ROTATE_IMAGE', default_value='False')},
        ],
        output='screen'
    )

    # Node 3: Touch Event Publisher Node
    touch_event_publisher_node = Node(
        package=package_name,
        executable='touch_event_publisher_node',
        name='touch_event_publisher_node',
        parameters=[
            {'nats_url': EnvironmentVariable('NATS_URL', default_value='nats://localhost:4222')},
            {'touch_detections_topic': EnvironmentVariable('TOUCH_DETECTIONS_TOPIC', default_value='/touch_detections')},
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
    

    # Event handler: when calibration node exits start detection nodes
    stop_calibration_and_start_detection = RegisterEventHandler(
        event_handler=OnProcessExit(
            target_action=screen_calibration_node,
            on_exit=[
                mediapipe_detection_node,
                touch_detection_node,
                touch_event_publisher_node,
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

