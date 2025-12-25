from setuptools import find_packages, setup
import os
from glob import glob

package_name = 'orbbec_hand_detection'

setup(
    name=package_name,
    version='0.0.1',
    packages=find_packages(exclude=['test']),
    data_files=[
        ('share/ament_index/resource_index/packages',
            ['resource/' + package_name]),
        ('share/' + package_name, ['package.xml']),
         # Include all launch files
        (os.path.join('share', package_name, 'launch'), glob('launch/*.py')),
        (os.path.join('share', package_name, 'rviz'), glob('rviz/*.rviz')),
        # Include message files
        #(os.path.join('share', package_name, 'msg'), glob('msg/*.msg')),
    ],
    install_requires=['setuptools'],
    zip_safe=True,
    maintainer='Dimitar Dimeski',
    maintainer_email='dimitar.dimeski23@gmail.com',
    description='Package for getting 3D coordinates of fingers pressing on an interactive table.',
    license='TODO: License declaration',
    tests_require=['pytest'],
    entry_points={
        'console_scripts': [
            'mediapipe_webcam = orbbec_hand_detection.mediapipe_webcam:main',
            'pointer_tip_depth = orbbec_hand_detection.pointer_tip_depth_publisher:main',
            'pointer_tip_depth_plane = orbbec_hand_detection.pointer_tip_depth_publisher_plane:main',
            'pointer_tip_depth_plane_depth_image = orbbec_hand_detection.pointer_tip_depth_publisher_plane_depth_image:main',
            'pointer_tip_depth_depth_image = orbbec_hand_detection.pointer_tip_depth_publisher_depth_image:main',
            'aruco_display_node = orbbec_hand_detection.show_aruco_display_node:main',
            'screen_calibration_node = orbbec_hand_detection.screen_calibration_node:main',
            # New 3-node pipeline
            'mediapipe_detection_node = orbbec_hand_detection.mediapipe_detection_node:main',
            'touch_detection_node = orbbec_hand_detection.touch_detection_node:main',
            'touch_event_publisher_node = orbbec_hand_detection.touch_event_publisher_node:main',
        ],
    },
)
