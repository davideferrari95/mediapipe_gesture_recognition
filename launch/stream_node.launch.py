import os
from typing import List
from ament_index_python.packages import get_package_share_directory

from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def create_node(config:List[str]):

    # Node - Parameters
    node_parameters = {

        # Webcam Parameters
        'webcam':    LaunchConfiguration('webcam'),
        'realsense': LaunchConfiguration('realsense'),

        # Enable Mediapipe Modules Parameters
        'enable_right_hand':     LaunchConfiguration('enable_right_hand'),
        'enable_left_hand':      LaunchConfiguration('enable_left_hand'),
        'enable_pose':           LaunchConfiguration('enable_pose'),
        'enable_face':           LaunchConfiguration('enable_face'),
        'enable_face_detection': LaunchConfiguration('enable_face_detection'),
        'enable_objectron':      LaunchConfiguration('enable_objectron'),

        # FaceMesh Connections Mode -> 0: Contours, 1: Tesselation
        'face_mesh_mode': LaunchConfiguration('face_mesh_mode'),

        # Available Models = 'Shoe', 'Chair', 'Cup', 'Camera'
        'objectron_model': LaunchConfiguration('objectron_model'),
    }

    # Python Node + Parameters
    stream_node = Node(
        package='mediapipe_gesture_recognition', executable='stream_node.py', name='mediapipe_stream_node',
        output='screen', output_format='{line}', emulate_tty=True, arguments=[('__log_level:=debug')],
        parameters=[node_parameters] + config,
    )

    # Return Node
    return stream_node

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Webcam Arguments
    webcam_arg    = DeclareLaunchArgument('webcam',    default_value='0')
    realsense_arg = DeclareLaunchArgument('realsense', default_value='false')

    # Enable Mediapipe Modules Args
    enable_right_hand_arg     = DeclareLaunchArgument('enable_right_hand',     default_value='true')
    enable_left_hand_arg      = DeclareLaunchArgument('enable_left_hand',      default_value='true')
    enable_pose_arg           = DeclareLaunchArgument('enable_pose',           default_value='true')
    enable_face_arg           = DeclareLaunchArgument('enable_face',           default_value='false')
    enable_face_detection_arg = DeclareLaunchArgument('enable_face_detection', default_value='false')
    enable_objectron_arg      = DeclareLaunchArgument('enable_objectron',      default_value='false')

    # FaceMesh Connections Mode -> 0: Contours, 1: Tesselation
    face_mesh_mode_arg = DeclareLaunchArgument('face_mesh_mode', default_value='0')

    # Available Models = 'Shoe', 'Chair', 'Cup', 'Camera'
    objectron_model_arg = DeclareLaunchArgument('objectron_model', default_value='Shoe')

    # Launch Description - Add Arguments
    launch_description.add_action(webcam_arg)
    launch_description.add_action(realsense_arg)
    launch_description.add_action(enable_right_hand_arg)
    launch_description.add_action(enable_left_hand_arg)
    launch_description.add_action(enable_pose_arg)
    launch_description.add_action(enable_face_arg)
    launch_description.add_action(enable_face_detection_arg)
    launch_description.add_action(enable_objectron_arg)
    launch_description.add_action(face_mesh_mode_arg)
    launch_description.add_action(objectron_model_arg)

    # Config File Path
    config = os.path.join(get_package_share_directory('mediapipe_gesture_recognition'), 'config','stream_node.yaml')

    # Launch Description - Add Nodes
    launch_description.add_action(create_node([config]))

    # Return Launch Description
    return launch_description
