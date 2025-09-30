from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory
import os

def generate_launch_description():

    rviz_config_path = os.path.join(
    get_package_share_directory('orbbec_hand_detection'), 
    'rviz', 
    'pointer_depth_config.rviz') 

    included_launch_path = os.path.join(
        get_package_share_directory('orbbec_camera'),
        'launch',
        'gemini2.launch.py'
    )

    return LaunchDescription([
        # Orbbec hand detection node
        Node(
            package='orbbec_hand_detection',
            executable='pointer_tip_depth',
            name='pointer_tip_depth'
        ),

        # Rviz node
        Node(
            package='rviz2',
            executable='rviz2',
            name='rviz2',
            arguments=['-d', rviz_config_path],
            output='screen'
        ),

        # Include another launch file
        IncludeLaunchDescription(
            PythonLaunchDescriptionSource(included_launch_path)
        )
    ])
