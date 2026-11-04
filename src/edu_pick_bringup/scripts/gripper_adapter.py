#!/usr/bin/env python3
"""RG2 그리퍼 어댑터 — 표준 ROS 액션을 레인보우 제어박스 명령으로 번역한다.

■ 왜 필요한가
  RG2 는 ros2_control 관절이 아니다. 레인보우 제어박스가 그리퍼를 물고 있고,
  제어박스 스크립트 명령 `gripper_macro …` 로만 움직인다.
  이 노드는 일반 팔 컨트롤러와 **같은 모양의 액션**(FollowJointTrajectory)을 열어 두고,
  목표 관절각을 받으면 제어박스 명령으로 바꿔 보낸다. 그래서 실습 노드는 그리퍼가
  무엇이든 똑같이 액션만 부르면 된다.

    /gripper_controller/follow_joint_trajectory  (액션 서버)  ← 실습 노드가 부른다
    /gripper_controller/joint_states             (토픽)       → 손가락 TF 표시용

■ 명령 형식 (실기 펜던트 스크립트에서 확인, 2026-10-29 · 2026-11-03 사진 재확인)
    gripper_macro 12,0,0,0,0,0,0,0,0,0           ← 기능 0 = 초기화 (첫 명령 전에 한 번)
    gripper_macro 12,0,1,0,<벌림%>,<힘%>,0,0,0,0 ← 기능 1 = 구동
    = 모델 12(OnRobot RG2) · 연결점 0(툴 플랜지) · 기능 · 0 · 벌림[%] · 힘[%] · 0×4
    펜던트 예: 12,0,1,0,48,54,… (벌림 48 % · 힘 54 %) / 12,0,1,0,100,100,… (최대 벌림)

■ 관절각 ↔ 벌림 %
    rg2_finger_joint1 = 0.0 rad → 0 %(닫힘),  0.6 rad → 100 %(열림)
    (모델 관절 상한은 1.18 이지만 실기 100 % 는 0.6 rad 에 해당 — 카메라 영상으로 확인)

■ ⚠ '잡혔는지'는 알 수 없다
  제어박스가 명령을 '접수'했다는 응답만 온다. 그래서 settle_sec 만큼 기다린 뒤 성공으로 본다.
  실제로 잡았는지는 실습 노드가 카메라로 다시 확인한다.

■ dry_run:=true (가짜 하드웨어 모드)
  명령을 보내지 않고 로그만 남긴다. 손가락 관절 상태는 그대로 바뀌므로 RViz 에서 움직임이 보인다.
"""
import rclpy
from rclpy.action import ActionServer
from rclpy.callback_groups import ReentrantCallbackGroup
from rclpy.executors import MultiThreadedExecutor
from rclpy.node import Node
from control_msgs.action import FollowJointTrajectory
from sensor_msgs.msg import JointState
import threading
import time

try:
    from rbpodo_msgs.srv import Eval
except ImportError:
    Eval = None

JOINT, MIMIC = "rg2_finger_joint1", "rg2_finger_joint2"


class GripperAdapter(Node):
    def __init__(self):
        super().__init__("gripper_adapter")
        self.declare_parameter("dry_run", False)
        self.declare_parameter("eval_service", "/rbpodo_hardware/eval")
        self.declare_parameter("joint_closed", 0.0)     # 0 %
        self.declare_parameter("joint_open", 0.6)       # 100 %
        self.declare_parameter("force_pct", 40)         # 힘 [%] — launch 가 scene.yaml gripper.force_pct 를 넘긴다
        self.declare_parameter("settle_sec", 1.5)       # 명령 후 기다리는 시간
        self.declare_parameter("cmd_move", "gripper_macro 12,0,1,0,{ratio_pct:.0f},{force_pct},0,0,0,0")
        self.declare_parameter("cmd_init", "gripper_macro 12,0,0,0,0,0,0,0,0,0")   # 빈 값이면 보내지 않음

        g = lambda n: self.get_parameter(n).value
        self.dry = bool(g("dry_run"))
        self.j_closed, self.j_open = float(g("joint_closed")), float(g("joint_open"))
        self._pos = self.j_open
        self._initialized = False
        self._cb = ReentrantCallbackGroup()

        self._cli = None
        if not self.dry:
            if Eval is None:
                raise RuntimeError("rbpodo_msgs 가 없습니다 — install.sh 를 실행했는지 확인하세요.")
            self._cli = self.create_client(Eval, g("eval_service"), callback_group=self._cb)

        self._js = self.create_publisher(JointState, "joint_states", 10)
        self.create_timer(1 / 30, self._pub_js)
        ActionServer(self, FollowJointTrajectory, "follow_joint_trajectory", self._on_goal,
                     callback_group=self._cb)
        self.declare_parameter("init_wait", 3.0)        # 초기화 뒤 기다리는 시간 [s]
        # 실기: 시작하자마자 초기화해 둔다. 첫 집기 직전에 초기화하면 끝나기 전에 명령이 와서
        #   그리퍼가 움직이지 않았다(2026-11-04 실기, 사과). 초기화 → 기다림 → 100 % 열기.
        if not self.dry:
            threading.Thread(target=self._startup_init, daemon=True).start()
        self.get_logger().info("그리퍼 어댑터 시작" + ("  [dry_run: 명령을 보내지 않음]" if self.dry else ""))

    def _startup_init(self):
        init = self.get_parameter("cmd_init").value
        if not init or not self._cli.wait_for_service(timeout_sec=60.0):
            return
        time.sleep(2.0)                                   # 팔 하드웨어가 자리 잡을 시간
        if self._send(init):
            time.sleep(float(self.get_parameter("init_wait").value))
            self._initialized = True
            self._send(self.get_parameter("cmd_move").value.format(
                ratio_pct=100.0, force_pct=int(self.get_parameter("force_pct").value)))
            self._pos = self.j_open
            self.get_logger().info("그리퍼 초기화 완료 → 100 % 열림")

    def _ratio(self, pos):
        r = (pos - self.j_closed) / (self.j_open - self.j_closed)
        return min(1.0, max(0.0, r))

    def _send(self, script):
        if self.dry:
            self.get_logger().info(f"[dry_run] {script}")
            return True
        if not self._cli.wait_for_service(timeout_sec=5.0):
            self.get_logger().error("제어박스 eval 서비스가 없습니다 — 실기 하드웨어가 활성인지 확인하세요.")
            return False
        req = Eval.Request()
        req.script = script
        fut = self._cli.call_async(req)
        t0 = time.time()
        while not fut.done() and time.time() - t0 < 5.0:
            time.sleep(0.02)
        res = fut.result() if fut.done() else None
        if res is None or not res.success:
            self.get_logger().error(f"제어박스가 명령을 받지 않음: {script}")
            return False
        self.get_logger().info(f"보냄: {script}")
        return True

    def _on_goal(self, gh):
        traj = gh.request.trajectory
        result = FollowJointTrajectory.Result()
        if JOINT not in traj.joint_names or not traj.points:
            gh.abort()
            result.error_code = FollowJointTrajectory.Result.INVALID_JOINTS
            return result
        target = float(traj.points[-1].positions[traj.joint_names.index(JOINT)])
        # 첫 구동 전에 초기화 명령을 한 번 (펜던트 스크립트와 같은 순서)
        init = self.get_parameter("cmd_init").value
        if init and not self._initialized:
            if not self._send(init):
                gh.abort()
                result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
                return result
            time.sleep(float(self.get_parameter("init_wait").value))
            self._initialized = True
        script = self.get_parameter("cmd_move").value.format(
            ratio_pct=self._ratio(target) * 100.0, force_pct=int(self.get_parameter("force_pct").value))
        if not self._send(script):
            gh.abort()
            result.error_code = FollowJointTrajectory.Result.PATH_TOLERANCE_VIOLATED
            return result
        time.sleep(float(self.get_parameter("settle_sec").value))
        self._pos = target
        gh.succeed()
        result.error_code = FollowJointTrajectory.Result.SUCCESSFUL
        return result

    def _pub_js(self):
        m = JointState()
        m.header.stamp = self.get_clock().now().to_msg()
        m.name = [JOINT, MIMIC]
        m.position = [self._pos, self._pos]
        self._js.publish(m)


def main():
    rclpy.init()
    node = GripperAdapter()
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    try:
        ex.spin()
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
