"""Pick and place 실습 — 전체 시스템 실행.

  ① 집에서 연습 (로봇·카메라 없이)
       ros2 launch edu_pick_bringup pick_place.launch.py
  ② 실습실 — 로봇에 접속하되 제어박스 Simulation 모드 (팔은 안 움직임, 카메라는 실제)
       ros2 launch edu_pick_bringup pick_place.launch.py mode:=real robot_ip:=192.168.35.100
  ③ 실습실 — 실제 구동 (강사 확인 후)
       ros2 launch edu_pick_bringup pick_place.launch.py mode:=real robot_ip:=192.168.35.100 cb_simulation:=false

  그다음 다른 터미널에서:  ros2 run edu_pick_task pick_place --step

■ 띄우는 노드
   robot_state_publisher   URDF → TF
   ros2_control_node       팔 하드웨어(가짜 또는 레인보우 제어박스) + arm_controller
   joint_state_publisher   팔 관절 + 그리퍼 관절을 합쳐 /joint_states
   gripper_adapter         RG2 (sim 에서는 명령을 보내지 않음)
   move_group              MoveIt (계획·실행)
   카메라                  sim: fake_camera(가상) / real: realsense2_camera(D405)
   object_detector         물체 인식 (sim·real 같은 코드)
   rviz2
"""
import os
import socket
import subprocess

from ament_index_python.packages import get_package_prefix, get_package_share_directory
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, IncludeLaunchDescription, OpaqueFunction
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node

# 레인보우 패치로 생긴 하드웨어 파라미터: (최소, 최대, 설명)
HW_PARAMS = {
    "servo_t2": (0.02, 0.2, "서보 유지시간[s] — 무선이면 0.1"),
    "speed_bar": (0.0, 1.0, "제어박스 속도바"),
    "max_joint_velocity": (0.0, 1.0, "관절 속도 상한[rad/s]"),
    "script_grace": (0.0, 10.0, "그리퍼 명령 뒤 서보 끊김 허용[s]"),
}
OWN_NODES = ("robot_state_publisher", "controller_manager", "joint_state_publisher", "gripper_adapter",
             "move_group", "object_detector", "fake_camera", "d405")


def _check_no_stale_nodes():
    """앞 실행에서 남은 노드가 있으면 멈춘다.
    남은 robot_state_publisher 가 옛 설정(/robot_description)을 계속 뿌리면 새로 띄운
    하드웨어가 **옛 설정으로 접속**한다(실기에서 실제로 겪은 일)."""
    try:
        out = subprocess.check_output(["ros2", "node", "list", "--no-daemon", "--spin-time", "1"],
                                      text=True, stderr=subprocess.DEVNULL, timeout=20)
    except Exception:
        return
    stale = sorted({l.strip() for l in out.splitlines() if l.strip().rsplit("/", 1)[-1] in OWN_NODES})
    if stale:
        raise RuntimeError("앞 실행의 노드가 남아 있습니다:\n  " + "\n  ".join(stale) +
                           "\n  모든 터미널의 launch 를 Ctrl+C 로 끄고 다시 실행하세요.")


def _preflight(ip):
    """제어박스 접속 확인. 안 하면 접속 실패 시 아무 에러 없이 무한 대기한다."""
    for port in (5000, 5001):
        try:
            with socket.create_connection((ip, port), timeout=3.0):
                pass
        except OSError as e:
            raise RuntimeError(f"제어박스 {ip}:{port} 에 접속할 수 없습니다 ({e})\n"
                               "  · 랜선·IP 설정을 확인하세요 (ping " + ip + ")\n"
                               "  · 로봇 없이 연습하려면 mode:=sim")


def _inject_hw_params(urdf, values):
    """레인보우 하드웨어 블록에 패치 파라미터를 넣는다.
    패치 안 된 드라이버는 모르는 파라미터를 **조용히 무시**하므로 설치본에 패치가 있는지 먼저 본다."""
    lib = os.path.join(get_package_prefix("rbpodo_hardware"), "lib", "librbpodo_hardware.so")
    blob = open(lib, "rb").read()
    tag = "<plugin>rbpodo_hardware/RBPodoHardwareInterface</plugin>"
    extra = ""
    for name, raw in values.items():
        lo, hi, desc = HW_PARAMS[name]
        v = float(raw)
        if not lo <= v <= hi:
            raise RuntimeError(f"{name}({desc}) 범위 밖: {v} (허용 {lo}~{hi})")
        if name.encode() not in blob:
            raise RuntimeError(f"rbpodo_hardware 에 '{name}' 패치가 없습니다 — install.sh 를 다시 실행하세요.")
        print(f"[pick_place] {name} = {v}  ({desc})")
        extra += f'\n      <param name="{name}">{v}</param>'
    return urdf.replace(tag, tag + extra)


def _scene(desc):
    import yaml
    with open(os.path.join(desc, "config", "scene.yaml")) as f:
        return yaml.safe_load(f)


def _setup(context):
    a = lambda n: context.launch_configurations[n]
    mode = a("mode")
    if mode not in ("sim", "real"):
        raise RuntimeError("mode 는 sim 또는 real 입니다.")
    real = mode == "real"
    cb_sim = a("cb_simulation").lower() in ("true", "1")
    desc = get_package_share_directory("edu_pick_description")
    bringup = get_package_share_directory("edu_pick_bringup")

    _check_no_stale_nodes()
    if real:
        _preflight(a("robot_ip"))

    urdf = subprocess.check_output([
        "xacro", os.path.join(desc, "urdf", "edu_pick.urdf.xacro"),
        f"use_fake_hardware:={'false' if real else 'true'}",
        f"robot_ip:={a('robot_ip')}", f"cb_simulation:={'true' if cb_sim else 'false'}"], text=True)
    if real:
        urdf = _inject_hw_params(urdf, {k: a(k) for k in HW_PARAMS if a(k)})
        print("[pick_place] 모드 = 실기 " + a("robot_ip") + " / 제어박스 " +
              ("Simulation (팔 안 움직임)" if cb_sim else "🔴 REAL — 팔이 실제로 움직입니다"))
    else:
        print("[pick_place] 모드 = sim (가짜 하드웨어 + 가상 카메라)")

    # 그리퍼: Simulation 모드에서도 제어박스가 그리퍼 명령은 실행할 수 있어(확인 안 됨) 기본은 '보내지 않음'
    g = a("gripper_dry_run").lower()
    dry = (not real or cb_sim) if g == "auto" else g in ("true", "1")
    print(f"[pick_place] 그리퍼 명령 = {'보내지 않음(dry_run)' if dry else '실제로 보냄'}")

    nodes = [
        Node(package="robot_state_publisher", executable="robot_state_publisher",
             parameters=[{"robot_description": urdf}]),
        Node(package="controller_manager", executable="ros2_control_node", output="both",
             parameters=[os.path.join(desc, "config", "controllers.yaml")],
             remappings=[("~/robot_description", "/robot_description"), ("joint_states", "arm/joint_states")]),
        Node(package="joint_state_publisher", executable="joint_state_publisher",
             parameters=[{"source_list": ["arm/joint_states", "gripper_controller/joint_states"], "rate": 30}]),
        Node(package="controller_manager", executable="spawner", arguments=["joint_state_broadcaster"]),
        Node(package="controller_manager", executable="spawner", arguments=["arm_controller"]),
        Node(package="edu_pick_bringup", executable="gripper_adapter.py", name="gripper_adapter",
             namespace="gripper_controller", output="screen",
             parameters=[{"dry_run": dry, "force_pct": int(_scene(desc)["gripper"].get("force_pct", 40)),
                          "settle_sec": 2.5,
                          "init_on_start": a("gripper_init").lower() in ("true", "1")}]),
        IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("edu_pick_moveit"), "launch", "move_group.launch.py"))),
        Node(package="edu_pick_perception", executable="object_detector", output="screen"),
    ]
    use_cam = a("camera").lower() in ("true", "1")
    if real and not use_cam:
        print("[pick_place] 카메라 끔(camera:=false) — 인식 결과가 나오지 않습니다")
    if real and use_cam:
        try:
            get_package_share_directory("realsense2_camera")
        except Exception:
            raise RuntimeError("RealSense 드라이버가 없습니다: sudo apt install ros-humble-realsense2-camera")
        rs = {"camera_name": "d405", "camera_namespace": "camera",
              "pointcloud.enable": "true", "pointcloud.ordered_pc": "true", "align_depth.enable": "true"}
        if a("camera_serial"):
            rs["serial_no"] = f"'{a('camera_serial')}'"
        nodes.append(IncludeLaunchDescription(PythonLaunchDescriptionSource(
            os.path.join(get_package_share_directory("realsense2_camera"), "launch", "rs_launch.py")),
            launch_arguments=rs.items()))
    elif not real:
        nodes.append(Node(package="edu_pick_perception", executable="fake_camera", output="screen"))
    if a("rviz").lower() in ("true", "1"):
        srdf = open(os.path.join(get_package_share_directory("edu_pick_moveit"), "config", "edu_pick.srdf")).read()
        nodes.append(Node(package="rviz2", executable="rviz2", output="log",
                          arguments=["-d", os.path.join(bringup, "config", "pick_place.rviz")],
                          parameters=[{"robot_description": urdf, "robot_description_semantic": srdf}]))
    return nodes


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument("mode", default_value="sim", description="sim = 로봇 없이 연습 / real = 실기"),
        DeclareLaunchArgument("robot_ip", default_value="192.168.35.100", description="제어박스 IP"),
        DeclareLaunchArgument("cb_simulation", default_value="true",
                              description="true = 제어박스 Simulation 모드(팔 안 움직임) / false = 🔴 실제 구동"),
        DeclareLaunchArgument("servo_t2", default_value="", description="무선 연결이면 0.1 (유선은 비워 둠)"),
        DeclareLaunchArgument("speed_bar", default_value="0.2"),
        DeclareLaunchArgument("max_joint_velocity", default_value="0.5",
                              description="관절 속도 상한[rad/s] — 이보다 빠른 명령은 하드웨어에서 잘린다"),
        DeclareLaunchArgument("gripper_dry_run", default_value="auto",
                              description="auto = 실제 구동(cb_simulation:=false)일 때만 그리퍼 명령을 보냄 / true / false"),
        DeclareLaunchArgument("script_grace", default_value="3.0",
                              description="그리퍼 명령 뒤 서보 끊김 허용[s] — 넓은 물체를 힘으로 쥘 때 1 s 로는 모자랐다(실기)"),
        DeclareLaunchArgument("gripper_init", default_value="true",
                              description="false = 시작할 때 그리퍼 초기화·열기를 하지 않음 (물체를 쥔 채 다시 접속할 때)"),
        DeclareLaunchArgument("camera", default_value="true", description="mode:=real 에서 D405 드라이버 실행 여부"),
        DeclareLaunchArgument("camera_serial", default_value="", description="카메라가 여러 대일 때 시리얼"),
        DeclareLaunchArgument("rviz", default_value="true"),
        OpaqueFunction(function=_setup),
    ])
