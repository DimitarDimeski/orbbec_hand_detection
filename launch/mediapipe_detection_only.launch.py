from launch import LaunchDescription
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
        'pointer_depth_config.rviz'
    )

    # MediaPipe Detection Node
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
            {'contrast': EnvironmentVariable('CONTRAST', default_value='1')},
            {'brightness': EnvironmentVariable('BRIGHTNESS', default_value='20')},
            {'debug': EnvironmentVariable('DEBUG', default_value='False')},
        ],
        output='screen'
    )

    # RViz node
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        arguments=['-d', rviz_config_path],
        output='screen'
    )

    return LaunchDescription([
        rviz_node,
        mediapipe_detection_node,
    ])

