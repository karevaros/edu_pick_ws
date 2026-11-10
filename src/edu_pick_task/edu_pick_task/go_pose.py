#!/usr/bin/env python3
"""자세 이동·관절 조그 도구 — 실기 첫 구동 확인, 수업 전 자세 맞추기용.

    ros2 run edu_pick_task go_pose home              # scene.yaml 의 이름 있는 자세로 (home / observe)
    ros2 run edu_pick_task go_pose observe
    ros2 run edu_pick_task go_pose --jog wrist3 5    # 관절 하나를 +5° (작게 움직여 보기)
    ros2 run edu_pick_task go_pose --show            # 지금 관절값(도) 출력만
    ros2 run edu_pick_task go_pose --grip 100        # 그리퍼 벌림 % (100 = 최대 10 cm · 0 = 닫힘)
    ros2 run edu_pick_task go_pose home --yes        # 확인 없이 실행 (가상 모드 전용으로 쓸 것)
    ros2 run edu_pick_task go_pose --robot omx home  # OMX-F (관절 이름 joint1~joint5)

기본은 '계획 → RViz 에 경로 표시 → Enter → 실행' (pick_place --step 과 같은 방식).
"""
import argparse
import math
import sys
import threading
import time

import rclpy
from rclpy.executors import MultiThreadedExecutor
from rclpy.signals import SignalHandlerOptions

from edu_pick_task.pick_place import PickPlace, StepFailed

MAX_JOG_DEG = 30.0


def main():
    ap = argparse.ArgumentParser(description="자세 이동·관절 조그")
    ap.add_argument("pose", nargs="?", help="home / observe")
    ap.add_argument("--jog", nargs=2, metavar=("JOINT", "DEG"), help="관절 하나를 DEG 도 만큼")
    ap.add_argument("--show", action="store_true", help="지금 관절값만 출력")
    ap.add_argument("--grip", type=float, metavar="PCT", help="그리퍼 벌림 %% (0~100)")
    ap.add_argument("--yes", action="store_true", help="확인 없이 실행")
    ap.add_argument("--robot", choices=["rb5", "omx"], default="rb5", help="로봇 (launch 의 robot:= 과 같게)")
    args, ros_args = ap.parse_known_args()
    if not (args.pose or args.jog or args.show or args.grip is not None):
        ap.error("자세 이름, --jog, --grip, --show 중 하나를 주세요")

    rclpy.init(args=[sys.argv[0]] + ros_args, signal_handler_options=SignalHandlerOptions.NO)
    node = PickPlace(step_mode=not args.yes, robot=args.robot)
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    code = 0
    try:
        t0 = time.time()
        while not all(j in node.joints for j in node.arm_joints):
            if time.time() - t0 > 20:
                raise StepFailed("/joint_states 가 들어오지 않음 — bringup 이 떠 있나요?")
            time.sleep(0.1)
        cur = [node.joints[j] for j in node.arm_joints]
        print("지금 관절[deg]: " + "  ".join(f"{j} {math.degrees(v):7.2f}" for j, v in zip(node.arm_joints, cur)))
        if args.show:
            return
        if args.grip is not None:
            pct = min(100.0, max(0.0, args.grip))
            print(f"그리퍼 벌림 {pct:.0f} %")
            node.gripper(node.width_to_joint(float(node.g["max_open"]) * pct / 100.0), f"그리퍼 {pct:.0f}%")
            print("완료 — 명령 전송(RG2 는 도달 여부를 알려 주지 않으니 눈으로 확인)")
            return
        if args.jog:
            name, deg = args.jog[0], float(args.jog[1])
            if name not in node.arm_joints:
                raise StepFailed(f"관절 이름은 {node.arm_joints} 중 하나")
            if abs(deg) > MAX_JOG_DEG:
                raise StepFailed(f"조그는 ±{MAX_JOG_DEG:.0f}° 이내만 (요청 {deg}°)")
            target = list(cur)
            target[node.arm_joints.index(name)] += math.radians(deg)
            what = f"{name} {deg:+.1f}°"
        else:
            poses = node.scene["poses"]
            if args.pose not in poses:
                raise StepFailed(f"scene.yaml poses 에 '{args.pose}' 없음 (있는 것: {list(poses)})")
            target = [math.radians(v) for v in poses[args.pose]]
            what = f"자세 '{args.pose}'"
        diff = [math.degrees(b - a) for a, b in zip(cur, target)]
        print(f"목표 {what} — 관절 변화[deg]: " + "  ".join(f"{j} {d:+7.2f}" for j, d in zip(node.arm_joints, diff)))
        node.move_joints(target, what)
        after = [node.joints[j] for j in node.arm_joints]
        err = max(abs(math.degrees(a - b)) for a, b in zip(after, target))
        print(f"완료 — 목표와 최대 차이 {err:.2f}°")
    except StepFailed as e:
        print(f"❌ 중단: {e}", flush=True)
        if node.active_goal is not None:
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
