import os
import sys
import yaml
from ament_index_python.packages import get_package_share_directory

sys.path.append(os.path.join(get_package_share_directory('tdt_vision'), 'launch'))

from launch_ros.descriptions import ComposableNode
from launch_ros.actions import ComposableNodeContainer, LoadComposableNodes
from launch.actions import TimerAction, Shutdown
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription, DeclareLaunchArgument
from launch.conditions import IfCondition
from launch.substitutions import LaunchConfiguration
from launch.launch_description_sources import PythonLaunchDescriptionSource
from ament_index_python.packages import get_package_share_directory


def _to_launch_bool(value, fallback):
    if isinstance(value, bool):
        return 'true' if value else 'false'
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in ('true', 'false'):
            return lowered
    return fallback


def _load_foxglove_default_from_config(default_value):
    config_path = os.environ.get(
        'TDT_RADAR_LAUNCH_CONFIG',
        '/workspace/T-DT_Radar/config/launch_params.yaml'
    )
    if not os.path.exists(config_path):
        return default_value

    try:
        with open(config_path, 'r', encoding='utf-8') as file:
            data = yaml.safe_load(file) or {}
    except Exception:
        return default_value

    foxglove_cfg = data.get('foxglove', {}) if isinstance(data, dict) else {}
    return _to_launch_bool(foxglove_cfg.get('radar_enable'), default_value)

def generate_launch_description():
    enable_foxglove_default = _load_foxglove_default_from_config('true')
    enable_foxglove = LaunchConfiguration('enable_foxglove')

    def get_camera_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='vision_camera_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_foxglove_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='foxglove_bridge_node',
            parameters=[ {'send_buffer_limit': 1000000000}],
            extra_arguments=[{'use_intra_process_comms': True}]
        )
  
    def get_radar_detect_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='radar_detect_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_radar_resolve_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='radar_resolve_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_debug_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='debug_node',
            extra_arguments=[{'use_intra_process_comms': True}]
        )
        
    def get_record_node(package, plugin):
        return ComposableNode(
            package=package,
            plugin=plugin,
            name='record_node',
            parameters=[ ] ,
            extra_arguments=[{'use_intra_process_comms': True}]
        )        

    def get_camera_detector_container(camera_node,radar_detect_node,radar_resolve_node,debug_node,record_node):
        return ComposableNodeContainer(
            name='camera_detector_container',
            namespace='',
            package='rclcpp_components',
            executable='component_container',
            composable_node_descriptions=[
                camera_node,
                radar_detect_node,
                radar_resolve_node,
                debug_node,
                record_node
            ],
            output='both',
            emulate_tty=True,
            on_exit=Shutdown(),
        )

    # 相机包不包含在内，需自行准备
    hik_camera_node = get_camera_node('tdt_vision', 'tdt_vision::NodeCamera')
    radar_detect_node = get_radar_detect_node('tdt_vision', 'tdt_radar::Detect')
    radar_resolve_node = get_radar_resolve_node('tdt_vision', 'tdt_radar::Resolve')
    foxglove_node = get_foxglove_node('foxglove_bridge', 'foxglove_bridge::FoxgloveBridge')
    tdt_debug_node = get_debug_node('tdt_vision', 'tdt_vision::NodeDebug')
    record_node = get_record_node('databag_tool', 'BagRecorderNode')

    cam_detector = get_camera_detector_container(hik_camera_node,radar_detect_node,radar_resolve_node,tdt_debug_node,record_node)
    load_foxglove = LoadComposableNodes(
        target_container='camera_detector_container',
        composable_node_descriptions=[foxglove_node],
        condition=IfCondition(enable_foxglove),
    )

    plugin_map_launch_cmd = IncludeLaunchDescription(
                PythonLaunchDescriptionSource([os.path.join(
                    get_package_share_directory('tdt_vision'), 'launch', 'map_server_launch.py')]),
             )
    return LaunchDescription([
            DeclareLaunchArgument(
                'enable_foxglove',
                default_value=enable_foxglove_default,
                description='Enable foxglove bridge component',
            ),
            cam_detector,
            load_foxglove,
            plugin_map_launch_cmd,
        ])