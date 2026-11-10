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
from launch.actions import DeclareLaunchArgument, OpaqueFunction
from launch_ros.actions import Node
from launch_ros.parameter_descriptions import ParameterValue
from launch.substitutions import Command


# 로봇별 파일: (URDF xacro, MoveIt 설정 폴더, SRDF)
ROBOTS = {
    "rb5": ("edu_pick.urdf.xacro", "", "edu_pick.srdf"),
    "omx": ("edu_pick_omx.urdf.xacro", "omx", "edu_pick_omx.srdf"),
}


def config_dir(robot):
    return os.path.join(get_package_share_directory("edu_pick_moveit"), "config", ROBOTS[robot][1])


def srdf_path(robot):
    return os.path.join(config_dir(robot), ROBOTS[robot][2])


def moveit_params(robot="rb5"):
    """move_group 에 넘기는 MoveIt 파라미터 묶음."""
    cfg = get_package_share_directory("edu_pick_moveit")
    sub = config_dir(robot)
    desc = get_package_share_directory("edu_pick_description")
    y = lambda n: yaml.safe_load(open(os.path.join(sub, n)))
    return [
        # 계획에는 기하만 필요 → 하드웨어 인자는 기본값으로 둬도 링크·관절이 같다
        {"robot_description": ParameterValue(
            Command(["xacro ", os.path.join(desc, "urdf", ROBOTS[robot][0])]), value_type=str)},
        {"robot_description_semantic": open(srdf_path(robot)).read()},
        {"robot_description_kinematics": y("kinematics.yaml")},
        {"robot_description_planning": y("joint_limits.yaml")},
        {"planning_pipelines": ["ompl"], "default_planning_pipeline": "ompl",
         "ompl": yaml.safe_load(open(os.path.join(cfg, "config", "ompl_planning.yaml")))},
    ]


def _setup(context):
    robot = context.launch_configurations["robot"]
    if robot not in ROBOTS:
        raise RuntimeError(f"robot 은 {' / '.join(ROBOTS)} 중 하나입니다 (지금: {robot})")
    ctrl = yaml.safe_load(open(os.path.join(config_dir(robot), "moveit_controllers.yaml")))
    return [
        Node(package="moveit_ros_move_group", executable="move_group", output="screen",
             parameters=moveit_params(robot) + [ctrl, {
                 "allow_trajectory_execution": True,
                 "publish_planning_scene": True, "publish_geometry_updates": True,
                 "publish_state_updates": True, "publish_transforms_updates": True}]),
    ]


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("robot", default_value="rb5", description="rb5 = RB5-850e / omx = ROBOTIS OMX-F"),
        OpaqueFunction(function=_setup),
    ])
