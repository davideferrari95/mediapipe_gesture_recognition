from launch import LaunchDescription
from launch_ros.actions import Node
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch.substitutions import LaunchConfiguration

def create_recognition_node(context):

    # Node - Parameters
    node_parameters = {
        'realsense': LaunchConfiguration('realsense'),
        'recognition_precision_probability': LaunchConfiguration('recognition_precision_probability'),
    }

    # Python Node + Parameters
    recognition_node = Node(
        package='mediapipe_gesture_recognition', executable='recognition_node.py', name='mediapipe_3D_recognition_node',
        output='screen', output_format='{line}', emulate_tty=True, arguments=[('__log_level:=debug')],
        parameters=[node_parameters],
    )

    # Python Node
    point_at_node = Node(
        package='mediapipe_gesture_recognition', executable='point_area_node.py', name='point_area_node',
        output='screen', output_format='{line}', emulate_tty=True, arguments=[('__log_level:=debug')],
    )

    # Realsense Argument
    realsense = LaunchConfiguration('realsense').perform(context)

    # Return Node
    return [recognition_node] if realsense else [recognition_node, point_at_node]

def generate_launch_description():

    # Launch Description
    launch_description = LaunchDescription()

    # Webcam Arguments
    realsense_arg = DeclareLaunchArgument('realsense', default_value='false')

    # Recognition Precision Arguments
    recognition_precision_probability_arg = DeclareLaunchArgument('recognition_precision_probability', default_value='0.8')

    # Launch Description - Add Arguments
    launch_description.add_action(realsense_arg)
    launch_description.add_action(recognition_precision_probability_arg)

    # Launch Description - Add Nodes
    launch_description.add_action(OpaqueFunction(function = create_recognition_node))

    # Return Launch Description
    return launch_description
