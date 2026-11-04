"""MoveIt move_group 만 띄운다 (edu_pick_bringup/pick_place.launch.py 가 포함해서 쓴다).

move_group 은 '계획(경로 찾기) + 실행(컨트롤러에 궤적 전달)'을 서비스·액션으로 제공하는 노드다.
실습 노드(pick_place)는 아래를 호출한다:
    /compute_ik               (서비스)  목표 자세 → 관절값
    /move_action              (액션)    관절 목표까지 충돌 없는 경로 계획 (pick_place 는 계획만 하고 아래로 실행)
    /compute_cartesian_path   (서비스)  손끝이 직선으로 움직이는 경로
    /execute_trajectory       (액션)    만든 궤적 실행
"""
import os
import yaml
from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


def moveit_params():
    """move_group 에 넘기는 MoveIt 파라미터 묶음."""
    cfg = get_package_share_directory("edu_pick_moveit")
    desc = get_package_share_directory("edu_pick_description")
    y = lambda n: yaml.safe_load(open(os.path.join(cfg, "config", n)))
    return [
        # 계획에는 기하만 필요 → 하드웨어 인자는 기본값으로 둬도 링크·관절이 같다
        {"robot_description": ParameterValue(
            Command(["xacro ", os.path.join(desc, "urdf", "edu_pick.urdf.xacro")]), value_type=str)},
        {"robot_description_semantic": open(os.path.join(cfg, "config", "edu_pick.srdf")).read()},
        {"robot_description_kinematics": y("kinematics.yaml")},
        {"robot_description_planning": y("joint_limits.yaml")},
        {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl", "ompl": y("ompl_planning.yaml")},
    ]


def generate_launch_description():
    cfg = get_package_share_directory("edu_pick_moveit")
    ctrl = yaml.safe_load(open(os.path.join(cfg, "config", "moveit_controllers.yaml")))
    return LaunchDescription([
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=moveit_params() + [ctrl, {
                 "allow_trajectory_execution": True,
                 "publish_planning_scene": True, "publish_geometry_updates": True,
                 "publish_state_updates": True, "publish_transforms_updates": True}]),
    ])
