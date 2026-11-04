#!/usr/bin/env python3
"""Pick and place 시나리오 노드 — 실습의 주인공.

    ros2 run edu_pick_task pick_place              # 연속 실행
    ros2 run edu_pick_task pick_place --step       # 움직이기 직전마다 멈춤: 경로를 RViz 로 확인 → Enter (실기 권장)
    ros2 run edu_pick_task pick_place --count 2    # 물체 2개를 차례로 옮김
    ros2 run edu_pick_task pick_place --shape round  # 원형(캔·병 윗면·공·사과)만 집는다

■ 시나리오 (12단계)
   1 초기위치      관절 목표  (MoveIt 계획)
   2 대기위치      관절 목표  — 카메라가 테이블을 내려다보는 자세
   3 물체인식      /edu_pick/objects 에서 집을 물체 고르기
   4 물체 위 이동  손끝 목표 자세 → 역기구학(IK) → 관절 목표   (그리퍼 미리 열기)
   5 하강          손끝 직선 이동 (Cartesian)
   6 잡기          그리퍼 닫기
   7 상승          손끝 직선 이동
   8 하역 위 이동  손끝 목표 자세 → IK → 관절 목표
   9 하강          손끝 직선 이동
  10 놓기          그리퍼 열기
  11 상승          손끝 직선 이동
  12 초기위치      (다음 물체가 있으면 2 대기위치로)
  + 결과 확인      대기위치에서 다시 찍어, 집은 물체가 원래 자리에서 사라졌는지 본다

■ 이 노드가 쓰는 ROS 인터페이스 (전부 MoveIt move_group 이 제공)
   /compute_ik  /move_action  /compute_cartesian_path  /execute_trajectory
   + /gripper_controller/follow_joint_trajectory (그리퍼 어댑터)

■ 손끝 프레임(tool_tip): z축 = 접근 방향, x축 = 손가락이 닫히는 방향.
   '위에서 아래로 집기' = tool_tip z축이 월드 -z 를 향하는 자세.
"""
import argparse
import math
import sys
import threading
import time

import numpy as np
import rclpy
import yaml
from rclpy.action import ActionClient
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from rclpy.time import Time
from tf2_ros import Buffer, TransformListener

from builtin_interfaces.msg import Duration
from control_msgs.action import FollowJointTrajectory
from geometry_msgs.msg import Pose, Quaternion
from moveit_msgs.action import ExecuteTrajectory, MoveGroup
from moveit_msgs.msg import Constraints, JointConstraint, MoveItErrorCodes, RobotState, RobotTrajectory
from moveit_msgs.msg import AttachedCollisionObject, CollisionObject, PlanningScene
from moveit_msgs.srv import ApplyPlanningScene, GetCartesianPath, GetPositionIK
from shape_msgs.msg import SolidPrimitive
from sensor_msgs.msg import JointState
from trajectory_msgs.msg import JointTrajectory, JointTrajectoryPoint

from edu_pick_msgs.msg import DetectedObjectArray

ARM_JOINTS = ["base", "shoulder", "elbow", "wrist1", "wrist2", "wrist3"]
GROUP, TIP, WORLD = "arm", "tool_tip", "world"
GRIPPER_OPEN, GRIPPER_CLOSED = 0.6, 0.0          # rg2_finger_joint1 [rad] (100 % / 0 %) — 물리적 끝값


class StepFailed(Exception):
    pass


def down_quat(yaw):
    """손끝이 아래(-z)를 보고, 닫히는 방향(x축)이 yaw 인 자세.  = Rz(yaw)·Rx(180°)"""
    return Quaternion(x=math.cos(yaw / 2), y=math.sin(yaw / 2), z=0.0, w=0.0)


class PickPlace(Node):
    def __init__(self, step_mode=False, shape=None):
        super().__init__("pick_place")
        from ament_index_python.packages import get_package_share_directory
        import os
        self.declare_parameter("scene_file", os.path.join(
            get_package_share_directory("edu_pick_description"), "config", "scene.yaml"))
        with open(self.get_parameter("scene_file").value) as f:
            self.scene = yaml.safe_load(f)
        self.m = self.scene["motion"]
        g = self.scene["gripper"]
        # '열기' 벌림 (scene.yaml gripper.open_pct). '잡기'는 물체마다 close_target() 으로 정한다
        self.grip_open = GRIPPER_OPEN * float(g.get("open_pct", 100)) / 100.0
        self.table_z = float(self.scene["table"]["top_z"])
        self.step_mode = step_mode
        self.shape_filter = shape        # None = 모양 무관 / "round" / "box"

        self.joints = {}
        self.active_goal = None
        self.objects, self.objects_time = None, 0.0
        self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(DetectedObjectArray, "/edu_pick/objects", self._on_objects, 10)
        self.tf = Buffer()
        TransformListener(self.tf, self)

        self.ik = self.create_client(GetPositionIK, "/compute_ik")
        self.apply_scene = self.create_client(ApplyPlanningScene, "/apply_planning_scene")
        self.obstacle_ids = set()     # MoveIt 장면에 넣어 둔 물체 이름
        self.placed = []              # 이미 옮겨 놓은 물체 (장애물로 계속 둔다)
        self.cart = self.create_client(GetCartesianPath, "/compute_cartesian_path")
        self.move = ActionClient(self, MoveGroup, "/move_action")
        self.exe = ActionClient(self, ExecuteTrajectory, "/execute_trajectory")
        self.grip = ActionClient(self, FollowJointTrajectory, "/gripper_controller/follow_joint_trajectory")
        # 정지용 (cancel_motion)
        from action_msgs.srv import CancelGoal
        from std_msgs.msg import String
        self.stop_pub = self.create_publisher(String, "/trajectory_execution_event", 10)
        from moveit_msgs.msg import DisplayTrajectory
        self.display_pub = self.create_publisher(DisplayTrajectory, "/display_planned_path", 10)
        self.ctrl_cancel = self.create_client(CancelGoal, "/arm_controller/follow_joint_trajectory/_action/cancel_goal")

    # ================================================================ 기본 도구
    def _on_js(self, msg):
        for n, p in zip(msg.name, msg.position):
            self.joints[n] = p

    def _on_objects(self, msg):
        self.objects, self.objects_time = msg, time.time()

    def wait(self, fut, timeout, what):
        """비동기 호출 결과 기다리기 (콜백은 다른 스레드의 executor 가 처리)."""
        t0 = time.time()
        while not fut.done():
            if time.time() - t0 > timeout:
                raise StepFailed(f"{what}: {timeout:.0f}초 안에 응답 없음")
            time.sleep(0.01)
        return fut.result()

    def run_action(self, client, goal, timeout, what):
        if not client.wait_for_server(timeout_sec=5.0):
            raise StepFailed(f"{what}: 액션 서버가 없습니다 ({client._action_name})")
        gh = self.wait(client.send_goal_async(goal), 10.0, what)
        if not gh.accepted:
            raise StepFailed(f"{what}: 목표가 거부됨")
        # ⚠ 정상으로 끝났을 때만 지운다. Ctrl+C·시간초과로 빠져나가면 남겨 둬야 cancel_motion 이
        #   이 목표를 취소할 수 있다(finally 에서 지웠더니 '취소했다'고 출력하고 팔은 끝까지 갔다 — 실측).
        self.active_goal = gh
        res = self.wait(gh.get_result_async(), timeout, what).result
        self.active_goal = None
        return res

    def cancel_motion(self):
        """진행 중인 동작 정지.

        ⚠ Humble 의 MoveIt /move_action 은 취소 요청을 받아 주지 않는다(실측: 취소 응답 GOAL_TERMINATED,
          팔은 목표까지 그대로 감). 그래서 두 경로로 동시에 멈춘다:
            ① /trajectory_execution_event 에 "stop"  — MoveIt 이 실행을 중단 (MoveGroupInterface.stop() 과 같은 방법)
            ② arm_controller 액션의 목표 전부 취소   — 컨트롤러가 직접 멈춤
          그리고 '응답'이 아니라 **관절값이 실제로 멈췄는지**로 확인한다."""
        from action_msgs.srv import CancelGoal
        from std_msgs.msg import String
        self.stop_pub.publish(String(data="stop"))
        req = CancelGoal.Request()                     # 빈 goal_info = 모든 목표 취소
        if self.ctrl_cancel.wait_for_service(timeout_sec=1.0):
            self.ctrl_cancel.call_async(req)
        if self.active_goal is not None:
            self.active_goal.cancel_goal_async()       # 받아 주는 서버(그리퍼 등)라면 이것도 먹는다
        # 정지 확인: 0.5초 간격 두 번 관절값이 거의 같으면 멈춘 것
        time.sleep(0.7)
        a = np.array([self.joints.get(j, 0.0) for j in ARM_JOINTS])
        time.sleep(0.5)
        b = np.array([self.joints.get(j, 0.0) for j in ARM_JOINTS])
        moved = float(np.max(np.abs(b - a)))
        if moved < 0.002:
            return "동작을 정지했습니다(관절 정지 확인)."
        return f"⚠ 정지 확인 실패 — 0.5초 동안 관절이 {math.degrees(moved):.1f}° 움직였습니다. 비상정지 버튼!"

    def current_state(self):
        rs = RobotState()
        rs.joint_state.name = list(self.joints.keys())
        rs.joint_state.position = list(self.joints.values())
        return rs

    def tip_pose(self):
        t = self.tf.lookup_transform(WORLD, TIP, Time())
        p = Pose()
        p.position.x, p.position.y, p.position.z = (t.transform.translation.x,
                                                    t.transform.translation.y, t.transform.translation.z)
        p.orientation = t.transform.rotation
        return p

    # ================================================================ 동작 단위
    def move_joints(self, target, what):
        """관절 목표로 이동 — MoveIt 이 충돌 없는 경로를 계획하고 실행한다."""
        goal = MoveGroup.Goal()
        r = goal.request
        r.group_name = GROUP
        r.num_planning_attempts = 5
        r.allowed_planning_time = 5.0
        r.max_velocity_scaling_factor = float(self.m["velocity_scaling"])
        r.max_acceleration_scaling_factor = float(self.m["velocity_scaling"])
        r.start_state.is_diff = True
        c = Constraints()
        for n, v in zip(ARM_JOINTS, target):
            c.joint_constraints.append(JointConstraint(joint_name=n, position=float(v), tolerance_above=1e-4,
                                                       tolerance_below=1e-4, weight=1.0))
        r.goal_constraints = [c]
        goal.planning_options.plan_only = True          # ① 계획만 → ② 확인 → ③ 실행 (execute)
        # 계획은 무작위 탐색(RRTConnect)이라 가끔 실패하거나, 사후 검사에서 무효로 걸린다 → 최대 3번
        for attempt in range(1, 4):
            res = self.run_action(self.move, goal, 60.0, what)
            if res.error_code.val == MoveItErrorCodes.SUCCESS:
                break
            self.say(f"   ⚠ {what}: 경로 계획 실패 (오류 코드 {res.error_code.val}) — 다시 계획 {attempt}/3")
        else:
            raise StepFailed(f"{what}: 경로 계획 3번 실패")
        self.execute(res.planned_trajectory, what)

    def execute(self, traj, what):
        """계획된 궤적을 RViz 에 보여 주고(단계 모드면 확인을 받고) 실행한다."""
        from moveit_msgs.msg import DisplayTrajectory
        jt = traj.joint_trajectory
        dur = jt.points[-1].time_from_start.sec + jt.points[-1].time_from_start.nanosec * 1e-9 if jt.points else 0.0
        self.display_pub.publish(DisplayTrajectory(trajectory_start=self.current_state(), trajectory=[traj]))
        self.confirm(f"경로 계획 완료 ({len(jt.points)}점, {dur:.1f}초) — RViz '계획된 경로'를 확인하세요")
        goal = ExecuteTrajectory.Goal()
        goal.trajectory = traj
        res = self.run_action(self.exe, goal, dur * 2 + 30.0, what)
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise StepFailed(f"{what}: 실행 실패 (오류 코드 {res.error_code.val})")

    def solve_ik(self, pose):
        """손끝 목표 자세 → 관절값. 그리퍼는 180° 대칭이라 yaw 와 yaw+180° 둘 다 풀어 덜 도는 쪽을 쓴다."""
        best = None
        cur = np.array([self.joints[j] for j in ARM_JOINTS])
        x, y = pose.orientation.x, pose.orientation.y
        yaw = 2 * math.atan2(y, x)
        for cand_yaw in (yaw, yaw + math.pi):
            req = GetPositionIK.Request()
            req.ik_request.group_name = GROUP
            req.ik_request.ik_link_name = TIP
            req.ik_request.robot_state = self.current_state()      # 현재 관절값에서 출발(가까운 해)
            req.ik_request.avoid_collisions = True
            req.ik_request.pose_stamped.header.frame_id = WORLD
            req.ik_request.pose_stamped.pose.position = pose.position
            req.ik_request.pose_stamped.pose.orientation = down_quat(cand_yaw)
            req.ik_request.timeout = Duration(sec=1)
            res = self.wait(self.ik.call_async(req), 5.0, "IK")
            if res.error_code.val != MoveItErrorCodes.SUCCESS:
                continue
            js = res.solution.joint_state
            q = np.array([js.position[js.name.index(j)] for j in ARM_JOINTS])
            cost = float(np.sum(np.abs(q - cur)))
            if best is None or cost < best[0]:
                best = (cost, q, cand_yaw)
        if best is None:
            raise StepFailed(f"IK 해 없음 — 손끝 ({pose.position.x:.3f}, {pose.position.y:.3f}, "
                             f"{pose.position.z:.3f}) 에 닿을 수 없거나 충돌")
        return best[1]

    def move_above(self, x, y, z, yaw, what):
        """(x, y, z) 위치로 손끝을 내려보게 이동."""
        p = Pose()
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation = down_quat(yaw)
        self.move_joints(self.solve_ik(p), what)

    def move_linear(self, dz, what):
        """손끝을 현재 자세 그대로 수직으로 dz[m] 만큼 직선 이동 (+ 위 / - 아래)."""
        start = self.tip_pose()
        goal = Pose()
        goal.position.x, goal.position.y = start.position.x, start.position.y
        goal.position.z = start.position.z + dz
        goal.orientation = start.orientation

        req = GetCartesianPath.Request()
        req.header.frame_id = WORLD
        req.group_name, req.link_name = GROUP, TIP
        req.start_state = self.current_state()
        req.waypoints = [goal]
        req.max_step = 0.005            # 5 mm 마다 IK
        req.jump_threshold = 0.0
        req.avoid_collisions = True
        res = self.wait(self.cart.call_async(req), 10.0, what)
        if res.fraction < 0.99:
            raise StepFailed(f"{what}: 직선 경로를 {res.fraction * 100:.0f}% 밖에 못 만듦")
        traj = res.solution.joint_trajectory
        q = np.array([p.positions for p in traj.points])
        if len(q) > 1 and np.max(np.abs(np.diff(q, axis=0))) > 0.15:
            raise StepFailed(f"{what}: 직선 경로 중 관절이 갑자기 크게 돈다(손목 뒤집힘) — 실행 안 함")
        self._retime(traj, abs(dz))
        self.execute(RobotTrajectory(joint_trajectory=traj), what)

    def _retime(self, traj, dist):
        """직선 경로에 시간을 붙인다 — 부드럽게 출발·정지(코사인 속도).
        평균 속도 linear_speed, 단 어느 관절도 0.35 rad/s 를 넘지 않게 전체 시간을 늘린다."""
        q = np.array([p.positions for p in traj.points])
        n = len(q)
        if n < 2:
            return
        T = max(dist / float(self.m["linear_speed"]), 1.0)
        s = np.linspace(0.0, 1.0, n)
        for _ in range(20):
            t = T / math.pi * np.arccos(1 - 2 * s)          # s(t) = (1 - cos(πt/T)) / 2 의 역함수
            dt = np.diff(t)
            v = np.max(np.abs(np.diff(q, axis=0)) / np.maximum(dt[:, None], 1e-6))
            if v <= 0.35:
                break
            T *= v / 0.35 * 1.05
        for p, ti in zip(traj.points, t):
            p.time_from_start = Duration(sec=int(ti), nanosec=int((ti % 1) * 1e9))
            p.velocities, p.accelerations = [], []

    def gripper(self, pos, what):
        self.confirm(f"그리퍼 {'열기' if pos > 0.3 else '닫기'}")
        goal = FollowJointTrajectory.Goal()
        goal.trajectory = JointTrajectory(joint_names=["rg2_finger_joint1"],
                                          points=[JointTrajectoryPoint(positions=[pos],
                                                                       time_from_start=Duration(sec=1))])
        res = self.run_action(self.grip, goal, 15.0, what)
        if res.error_code != FollowJointTrajectory.Result.SUCCESSFUL:
            raise StepFailed(f"{what}: 그리퍼 실패 (오류 코드 {res.error_code})")

    # ================================================================ MoveIt 장면 (장애물)
    # MoveIt 은 URDF 에 있는 것(테이블·받침대·로봇 자신)만 안다. 카메라로 본 물체·이미 놓은 물체는
    # 직접 '장면(planning scene)'에 넣어 줘야 피해 간다. 쥔 물체는 그리퍼에 '붙여서' 함께 움직인다.
    PAD = 0.01   # 장애물 여유 [m]

    @staticmethod
    def _primitive(shape, width, length, height, pad):
        sp = SolidPrimitive()
        if shape == "round":
            sp.type, sp.dimensions = SolidPrimitive.CYLINDER, [height + pad, width / 2 + pad]
        else:
            sp.type, sp.dimensions = SolidPrimitive.BOX, [width + 2 * pad, length + 2 * pad, height + pad]
        return sp

    def _apply(self, ps, what):
        if not self.apply_scene.wait_for_service(timeout_sec=5.0):
            raise StepFailed("/apply_planning_scene 서비스 없음")
        res = self.wait(self.apply_scene.call_async(ApplyPlanningScene.Request(scene=ps)), 10.0, what)
        if not res.success:
            raise StepFailed(f"{what}: MoveIt 장면 반영 실패")

    def set_obstacles(self, items):
        """장애물 목록으로 장면을 바꾼다. items: dict(id, shape, x, y, bottom, width, length, height, yaw)"""
        ps = PlanningScene(is_diff=True)
        ps.robot_state.is_diff = True
        for oid in self.obstacle_ids:
            ps.world.collision_objects.append(CollisionObject(id=oid, operation=CollisionObject.REMOVE))
        for it in items:
            co = CollisionObject(id=it["id"], operation=CollisionObject.ADD)
            co.header.frame_id = WORLD
            co.primitives = [self._primitive(it["shape"], it["width"], it["length"], it["height"], self.PAD)]
            p = Pose()
            p.position.x, p.position.y = float(it["x"]), float(it["y"])
            p.position.z = float(it["bottom"]) + (it["height"] + self.PAD) / 2
            yaw = 0.0 if it["shape"] == "round" or math.isnan(it["yaw"]) else it["yaw"]
            p.orientation.z, p.orientation.w = math.sin(yaw / 2), math.cos(yaw / 2)
            co.primitive_poses = [p]
            ps.world.collision_objects.append(co)
        self._apply(ps, "장애물 갱신")
        self.obstacle_ids = {it["id"] for it in items}

    @staticmethod
    def as_item(o, oid=None):
        return {"id": oid or f"obstacle_{o.id}", "shape": o.shape, "x": o.top_center.x, "y": o.top_center.y,
                "bottom": o.top_center.z - o.height, "width": o.width, "length": o.length,
                "height": o.height, "yaw": o.yaw}

    def attach_held(self, obj, depth):
        """쥔 물체를 손끝(tool_tip)에 붙인다. depth = 손끝이 물체 윗면보다 내려간 깊이.
        tool_tip z축 = 접근(아래) 방향 → 물체 윗면은 z = -depth, 중심은 z = -depth + 높이/2"""
        a = AttachedCollisionObject(link_name=TIP,
                                    touch_links=["rg2_leftfinger", "rg2_rightfinger", "rg2_hand", TIP])
        a.object.id, a.object.operation = "held_object", CollisionObject.ADD
        a.object.header.frame_id = TIP
        a.object.primitives = [self._primitive(obj.shape, obj.width, obj.length, obj.height, 0.003)]
        p = Pose()
        p.position.z = -depth + obj.height / 2
        p.orientation.w = 1.0
        a.object.primitive_poses = [p]
        ps = PlanningScene(is_diff=True)
        ps.robot_state.is_diff = True
        ps.robot_state.attached_collision_objects = [a]
        self._apply(ps, "쥔 물체 붙이기")

    def detach_held(self, strict=True):
        """쥔 물체를 뗀다. strict=False: 붙은 게 없어도 괜찮다(시작할 때 정리용 —
        없는 물체를 지우라고 하면 MoveIt 은 '실패'를 돌려준다)."""
        a = AttachedCollisionObject(link_name=TIP)
        a.object.id, a.object.operation = "held_object", CollisionObject.REMOVE
        ps = PlanningScene(is_diff=True)
        ps.robot_state.is_diff = True
        ps.robot_state.attached_collision_objects = [a]
        try:
            self._apply(ps, "쥔 물체 떼기")
        except StepFailed:
            if strict:
                raise
            return
        # MoveIt 은 떼어 낸 물체를 그 자리에 '세상 물체'로 남긴다 → 지운다(놓은 물체는 placed 로 따로 관리)
        ps = PlanningScene(is_diff=True)
        ps.robot_state.is_diff = True
        ps.world.collision_objects = [CollisionObject(id="held_object", operation=CollisionObject.REMOVE)]
        try:
            self._apply(ps, "떼어 낸 물체 정리")
        except StepFailed:
            pass

    def close_target(self, width):
        """잡기 목표 관절값: 물체 폭보다 close_squeeze 만큼 좁게 → 닿는 순간부터 설정한 힘으로 쥔다."""
        g = self.scene["gripper"]
        w = max(0.0, width - float(g.get("close_squeeze", 0.015)))
        pct = min(100.0, 100.0 * w / float(g["max_open"]))
        return GRIPPER_OPEN * pct / 100.0, pct

    # ================================================================ 인식
    def detect(self, timeout=5.0):
        """대기위치에 멈춘 뒤 들어온 인식 결과만 쓴다(이동 중에 찍힌 결과는 버림)."""
        t_req = time.time() + 1.0                   # 멈춘 뒤 1초 이후 결과
        while time.time() < t_req + timeout:
            if self.objects is not None and self.objects_time > t_req:
                return list(self.objects.objects)
            time.sleep(0.05)
        raise StepFailed("인식 결과가 들어오지 않음 — /edu_pick/objects 를 확인하세요")

    def place_slots(self, n):
        """k 번째 물체를 놓을 xy — 하역 위치에서 step_xy 씩 옆으로."""
        pl = self.scene["place"]
        return [(pl["xy"][0] + k * pl["step_xy"][0], pl["xy"][1] + k * pl["step_xy"][1]) for k in range(n)]

    def choose(self, objects, slots):
        """집을 물체 고르기: 하역 구역 밖 · 집을 수 있는 것 중 테이블 중앙(roi 중심)에 가장 가까운 것."""
        roi = self.scene["detection"]
        cx, cy = sum(roi["roi_x"]) / 2, sum(roi["roi_y"]) / 2
        keep_out = float(self.scene["place"]["keep_out"])
        ok = []
        for o in objects:
            in_place = any(math.hypot(o.top_center.x - sx, o.top_center.y - sy) < keep_out for sx, sy in slots)
            why = "하역 구역(이미 옮긴 물체)" if in_place else o.note
            self.say(f"   · {o.id}: {o.shape:5s} 폭 {o.width * 100:4.1f} cm  높이 {o.height * 100:4.1f} cm  "
                     f"위치 ({o.top_center.x:.3f}, {o.top_center.y:.3f})" + (f"  ✕ {why}" if why else ""))
            if o.graspable and not in_place and (self.shape_filter is None or o.shape == self.shape_filter):
                ok.append(o)
        if not ok:
            return None
        return min(ok, key=lambda o: math.hypot(o.top_center.x - cx, o.top_center.y - cy))

    # ================================================================ 시나리오
    def say(self, text):
        print(text, flush=True)

    def step(self, no, name):
        self.say(f"\n[{no:>3}] {name}")

    def confirm(self, text):
        """단계 모드: 실제로 움직이기 직전에 멈추고 Enter 를 기다린다."""
        if self.step_mode:
            try:
                input(f"     ↳ {text}\n       Enter = 실행 / Ctrl+C = 중단 ")
            except EOFError:
                raise StepFailed("입력이 닫혀 실행하지 않고 중단합니다")

    def run(self, count=1):
        poses = {k: [math.radians(v) for v in vals] for k, vals in self.scene["poses"].items()}
        slots = self.place_slots(count)
        m = self.m

        self.say("MoveIt·컨트롤러·관절 상태 기다리는 중 …")
        t0 = time.time()
        while not all(j in self.joints for j in ARM_JOINTS):
            if time.time() - t0 > 20:
                raise StepFailed("/joint_states 가 들어오지 않음 — bringup 이 떠 있나요?")
            time.sleep(0.1)
        for c, n in ((self.ik, "/compute_ik"), (self.cart, "/compute_cartesian_path")):
            if not c.wait_for_service(timeout_sec=20.0):
                raise StepFailed(f"{n} 서비스 없음 — move_group 이 떠 있나요?")

        self.detach_held(strict=False)     # 앞 실행이 남긴 장면 정리
        self.set_obstacles([])
        self.step(1, "초기위치로 이동")
        self.move_joints(poses["home"], "초기위치")

        done = 0
        for k in range(count):
            self.step(2, "대기위치로 이동")
            self.move_joints(poses["observe"], "대기위치")

            self.step(3, "물체인식")
            seen = self.detect()
            obj = self.choose(seen, slots)
            # 고른 물체 말고는 전부 장애물로: 인식한 다른 물체 + 이미 옮겨 놓은 물체
            others = [self.as_item(o) for o in seen if obj is None or o.id != obj.id]
            self.set_obstacles(others + self.placed)
            self.say(f"   → MoveIt 장면 장애물 {len(others) + len(self.placed)}개 (인식 {len(others)} · 옮겨 놓은 것 {len(self.placed)})")
            if obj is None:
                self.say("   → 집을 수 있는 물체가 없습니다. 종료합니다.")
                break
            yaw = 0.0 if math.isnan(obj.yaw) else obj.yaw
            if math.isnan(obj.yaw):                 # 원형: 지금 손목 방향 그대로
                q = self.tip_pose().orientation
                yaw = 2 * math.atan2(q.y, q.x)
            # 잡는 높이(손끝): 키 큰 물체는 윗면에서 grip_depth 아래.
            # 낮은 물체(높이 < 폭×1.3 — 공·사과·누운 병·납작한 박스)는 가운데 높이:
            #   누운 병을 축보다 위에서 잡으면 들 때 미끄러진다
            low = obj.height < 1.3 * obj.width
            grip_z = obj.top_center.z - (obj.height * 0.5 if low else float(m["grip_depth"]))
            # RG2 는 닫을수록 손끝이 앞(아래)으로 나온다 — 실측 2점(10 cm 벌림 20.5 cm · 0 cm 24 cm) 사이 직선 근사.
            # 물체 폭까지 닫혔을 때의 손끝도 테이블 위 min_tip_clearance 이상이어야 한다.
            g = self.scene["gripper"]
            advance = (g["tip_closed"] - g["tip_open"]) * (1.0 - min(obj.width / g["max_open"], 1.0))
            grip_z = max(grip_z, self.table_z + float(m["min_tip_clearance"]) + advance)
            above_z = obj.top_center.z + float(m["approach_height"])
            self.say(f"   → 선택: {obj.id} ({obj.shape}) · 집기 방향 {math.degrees(yaw):.0f}° · "
                     f"손끝 높이 {grip_z:.3f} m (닫으면 {advance * 100:.1f} cm 더 내려감 → 테이블 위 "
                     f"{(grip_z - advance - self.table_z) * 100:.1f} cm)")

            self.step(4, "물체 위로 이동 (그리퍼 열기)")
            self.gripper(self.grip_open, "그리퍼 열기")
            self.move_above(obj.top_center.x, obj.top_center.y, above_z, yaw, "물체 위 이동")

            self.step(5, f"하강 {(above_z - grip_z) * 100:.1f} cm")
            self.move_linear(grip_z - above_z, "하강")

            close_pos, close_pct = self.close_target(obj.width)
            self.step(6, f"잡기 (물체 폭 {obj.width * 100:.1f} cm → 그리퍼 목표 {close_pct:.0f} %)")
            self.gripper(close_pos, "잡기")
            self.attach_held(obj, obj.top_center.z - grip_z)

            self.step(7, "상승")
            self.move_linear(above_z - grip_z, "상승")

            # 놓을 때 손끝 높이 = 잡을 때 손끝이 물체 바닥에서 떨어진 높이 + 여유
            release_z = self.table_z + (grip_z - (obj.top_center.z - obj.height)) + float(m["release_gap"])
            place_above = release_z + float(m["approach_height"])
            self.step(8, "하역 위치 위로 이동")
            place_xy = slots[k]
            self.say(f"   → 하역 위치 ({place_xy[0]:.2f}, {place_xy[1]:.2f})")
            self.move_above(place_xy[0], place_xy[1], place_above, yaw, "하역 위치 이동")

            self.step(9, f"하강 {(place_above - release_z) * 100:.1f} cm")
            self.move_linear(release_z - place_above, "하강")

            self.step(10, "놓기")
            self.gripper(self.grip_open, "놓기")
            self.detach_held()

            self.step(11, "상승")
            self.move_linear(place_above - release_z, "상승")
            # 놓은 물체는 손이 빠져나온 '뒤에' 장애물로 등록한다(손가락이 둘러싼 채로 넣으면 시작부터 충돌)
            it = self.as_item(obj, oid=f"placed_{k}")
            it.update(x=place_xy[0], y=place_xy[1], bottom=self.table_z)
            self.placed.append(it)
            self.set_obstacles(list(self.placed))

            self.step("11+", "결과 확인 (대기위치에서 다시 인식)")
            self.move_joints(poses["observe"], "대기위치")
            left = [o for o in self.detect()
                    if math.hypot(o.top_center.x - obj.top_center.x, o.top_center.y - obj.top_center.y) < 0.03]
            if left:
                self.say("   ✕ 물체가 원래 자리에 그대로 있습니다 — 잡기 실패")
            else:
                self.say("   ✓ 원래 자리에서 사라졌습니다 — 옮기기 성공")
                done += 1

        self.step(12, "초기위치로 이동")
        self.move_joints(poses["home"], "초기위치")
        self.set_obstacles([])
        self.say(f"\n완료 — {done}개 옮김")


def main():
    ap = argparse.ArgumentParser(description="Pick and place 시나리오")
    ap.add_argument("--step", action="store_true", help="단계마다 Enter 로 진행")
    ap.add_argument("--count", type=int, default=1, help="옮길 물체 개수")
    ap.add_argument("--shape", choices=["round", "box"], help="이 모양만 집는다")
    args, ros_args = ap.parse_known_args()

    # Ctrl+C 를 rclpy 가 가로채면 통신이 먼저 끊겨 '정지 명령'을 못 보낸다 → 직접 처리
    from rclpy.signals import SignalHandlerOptions
    rclpy.init(args=[sys.argv[0]] + ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = PickPlace(step_mode=args.step, shape=args.shape)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    code = 0
    try:
        node.run(args.count)
    except StepFailed as e:
        print(f"\n❌ 중단: {e}", flush=True)
        if node.active_goal is not None:          # 시간초과 등으로 빠져나왔으면 남은 동작도 멈춘다
            print("   " + node.cancel_motion(), flush=True)
        code = 1
    except KeyboardInterrupt:
        print("\n중단(Ctrl+C) — " + node.cancel_motion(), flush=True)
        code = 130
    finally:
        ex.shutdown()
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()
    sys.exit(code)


if __name__ == "__main__":
    main()
