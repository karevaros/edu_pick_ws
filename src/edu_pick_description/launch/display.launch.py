"""로봇 모델만 RViz 로 보기 (슬라이더로 관절 움직이기) — 실습 1단계 'URDF 와 TF'.

    ros2 launch edu_pick_description display.launch.py
"""
import os
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def generate_launch_description():
    share = get_package_share_directory("edu_pick_description")
    urdf = ParameterValue(Command(["xacro ", os.path.join(share, "urdf", "edu_pick.urdf.xacro")]),
                          value_type=str)
    return LaunchDescription([
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="joint_state_publisher_gui", executable="joint_state_publisher_gui"),
        Node(package="rviz2", executable="rviz2",
             arguments=["-d", os.path.join(share, "rviz", "display.rviz")]),
    ])
