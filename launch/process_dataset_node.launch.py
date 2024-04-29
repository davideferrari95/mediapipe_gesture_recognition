from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration

def create_node():

    # Node Parameters
    node_parameters = {
        'enable_right_hand': LaunchConfiguration('enable_right_hand'),
        'enable_left_hand':  LaunchConfiguration('enable_left_hand'),
        'enable_pose':       LaunchConfiguration('enable_pose'),
        'enable_face':       LaunchConfiguration('enable_face'),
        'debug':             LaunchConfiguration('debug'),
    }

    # Python Node + Parameters
    example_node = Node(
        package='mediapipe_gesture_recognition', executable='process_dataset_node.py', name='process_dataset_node',
        output='screen', output_format='{line}', emulate_tty=True,
        parameters=[node_parameters],
    )

    return example_node

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Enable Mediapipe Modules Args
    enable_right_hand_arg = DeclareLaunchArgument('enable_right_hand', default_value='true')
    enable_left_hand_arg  = DeclareLaunchArgument('enable_left_hand',  default_value='true')
    enable_pose_arg       = DeclareLaunchArgument('enable_pose',       default_value='true')
    enable_face_arg       = DeclareLaunchArgument('enable_face',       default_value='false')
    debug_arg             = DeclareLaunchArgument('debug',             default_value='true')

    # Launch Description - Add Arguments
    launch_description.add_action(enable_right_hand_arg)
    launch_description.add_action(enable_left_hand_arg)
    launch_description.add_action(enable_pose_arg)
    launch_description.add_action(enable_face_arg)
    launch_description.add_action(debug_arg)

    # Launch Description - Add Nodes
    launch_description.add_action(create_node())

    # Return Launch Description
    return launch_description
