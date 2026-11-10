#!/usr/bin/env python3
"""[강사용 도구] OMX 손목 카메라 보정 — 자 없이, 팔을 조금씩 움직여 카메라 위치와 화각을 구한다.

필요한 것: OMX 판 bringup 이 실기로 떠 있을 것(카메라 드라이버는 끈 채로)
    ros2 launch edu_pick_bringup pick_place.launch.py robot:=omx mode:=real camera:=false
    테이블 위 대기위치 아래쪽에 빨간 둥근 물체(토마토·공) 하나 — 공 모양이어야 한다(높이 = 지름으로 본다)

    python3 tools/omx_cam_calib.py            # 측정 + 계산 + 결과 출력 (scene_omx.yaml 은 고치지 않음)
    python3 tools/omx_cam_calib.py --x 0.21   # 손끝 시작 위치 지정 (공이 화면 가장자리에 걸리면)

■ 원리
  손끝을 위에서 아래로 향한 채 ①좌우·앞뒤 2 cm 옮기기 ②2 cm 들기 ③손목만 ±25° 돌리기 를 하면서
  공의 영상 위치·크기를 잰다. 공은 그대로이고 손끝 위치(TF)는 정확히 아므로,
  '카메라가 손끝에 대해 어디·어느 방향에 붙어 있고 초점거리가 얼마면 이 영상들이 나오는가'를
  최소제곱으로 푼다(바늘구멍 카메라 근사 — 광각 왜곡은 무시하므로 공을 화면 가운데 근처에 둘 것).

  미지수: 초점거리 f · 카메라 위치(손끝 기준 x, y, z) · 카메라 방향(손끝 축 둘레 회전, 기울기 2) · 공 위치 x, y · 공 지름
"""
import math
import os
import sys
import threading
import time

import cv2
import numpy as np
import rclpy
import yaml
from rclpy.executors import MultiThreadedExecutor
from rclpy.time import Time
from scipy.optimize import least_squares

from builtin_interfaces.msg import Duration
from geometry_msgs.msg import Pose
from moveit_msgs.msg import MoveItErrorCodes
from moveit_msgs.srv import GetPositionIK

from edu_pick_task.pick_place import PickPlace, down_quat, StepFailed

STEP, LIFT, ROLL = 0.02, 0.02, math.radians(25)


def quat_mat(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rot(axis, a):
    c, s = math.cos(a), math.sin(a)
    if axis == "x":
        return np.array([[1, 0, 0], [0, c, -s], [0, s, c]])
    if axis == "y":
        return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])


class Calib:
    def __init__(self, node, device, hsv):
        self.n, self.hsv = node, hsv
        self.cap = cv2.VideoCapture(device, cv2.CAP_V4L2)
        self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        if not self.cap.isOpened():
            raise SystemExit(f"카메라를 열 수 없음: {device} (usb_cam 노드가 쓰고 있으면 camera:=false 로 띄울 것)")
        self.prev = None

    def frame(self, settle=1.2):
        t = time.time()
        while time.time() - t < settle:          # 버퍼에 쌓인 옛 영상 버리기 + 자동 노출 안정
            ok, f = self.cap.read()
        if not ok:
            raise SystemExit("카메라 영상 없음")
        return f

    def blob(self, img):
        hsv = cv2.cvtColor(cv2.GaussianBlur(img, (5, 5), 0), cv2.COLOR_BGR2HSV)
        m = np.zeros(img.shape[:2], np.uint8)
        for lo, hi in self.hsv:
            m |= cv2.inRange(hsv, np.array(lo), np.array(hi))
        m = cv2.morphologyEx(m, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        H, W = m.shape
        cand = []
        for c in cs:
            x, y, w, h = cv2.boundingRect(c)
            if cv2.contourArea(c) < 800 or x <= 1 or y <= 1 or x + w >= W - 1 or y + h >= H - 1:
                continue                          # 너무 작거나 화면 가장자리에 잘린 덩어리
            M = cv2.moments(c)
            u, v = M["m10"] / M["m00"], M["m01"] / M["m00"]
            d = 2 * math.sqrt(cv2.contourArea(c) / math.pi)   # 같은 면적 원의 지름 [px]
            ref = self.prev if self.prev is not None else (W / 2, H / 2)
            cand.append((math.hypot(u - ref[0], v - ref[1]), u, v, d))
        if not cand:
            return None
        _, u, v, d = min(cand)
        self.prev = (u, v)
        return u, v, d

    def ik_exact(self, x, y, z, yaw):
        """yaw 를 그대로(180° 뒤집지 않고) 푸는 IK — 카메라 방향이 바뀌면 안 된다."""
        req = GetPositionIK.Request()
        r = req.ik_request
        r.group_name, r.ik_link_name, r.avoid_collisions = "arm", "tool_tip", True
        r.robot_state = self.n.current_state()
        r.pose_stamped.header.frame_id = "world"
        p = r.pose_stamped.pose
        p.position.x, p.position.y, p.position.z = float(x), float(y), float(z)
        p.orientation = down_quat(yaw)
        r.timeout = Duration(sec=1)
        res = self.n.wait(self.n.ik.call_async(req), 5.0, "IK")
        if res.error_code.val != MoveItErrorCodes.SUCCESS:
            raise StepFailed(f"IK 실패 ({x:.3f}, {y:.3f}, {z:.3f}, yaw {math.degrees(yaw):.0f}°)")
        js = res.solution.joint_state
        return [js.position[js.name.index(j)] for j in self.n.arm_joints]

    def tool_pose(self):
        t = self.n.tf.lookup_transform("world", "tool_tip", Time())
        tr, q = t.transform.translation, t.transform.rotation
        return np.array([tr.x, tr.y, tr.z]), quat_mat((q.x, q.y, q.z, q.w))


def project(params, Tw, Rw, W, H):
    """params → (u, v, 지름 px). 카메라는 손끝(tool_tip) 프레임에 고정."""
    f, cx_, cy_, cz_, rz, rx, ry, X, Y, D = params
    R_tc = rot("z", rz) @ rot("x", rx) @ rot("y", ry)            # 손끝 → 광학 (광학 z ≈ 손끝 z = 접근 방향)
    c_w = Tw + Rw @ np.array([cx_, cy_, cz_])                    # 카메라 위치 (world)
    R_wc = Rw @ R_tc
    P = R_wc.T @ (np.array([X, Y, D / 2]) - c_w)                 # 공 중심 → 광학 좌표
    return np.array([W / 2 + f * P[0] / P[2], H / 2 + f * P[1] / P[2], f * D / P[2]])


def main():
    rclpy.init()
    node = PickPlace(robot="omx")
    ex = MultiThreadedExecutor()
    ex.add_node(node)
    threading.Thread(target=ex.spin, daemon=True).start()
    scene = node.scene
    red = next(c for c in scene["detection"]["colors"] if c["name"] == "red")
    hsv = [red["hsv"]] + ([red["hsv2"]] if "hsv2" in red else [])
    cal = Calib(node, scene["camera"]["device"], hsv)
    while not all(j in node.joints for j in node.arm_joints):
        time.sleep(0.1)
    time.sleep(1.0)

    T0, _ = cal.tool_pose()
    x0, y0, z0, yaw0 = T0[0], T0[1], T0[2], math.pi / 2
    # 시작 위치 바꾸기: --x 0.21 --y 0.0 --z 0.11 (공이 화면 가운데 오게)
    args = dict(zip(sys.argv[1::2], sys.argv[2::2]))
    x0 = float(args.get("--x", x0))
    y0 = float(args.get("--y", y0))
    z0 = float(args.get("--z", z0))
    print(f"기준 손끝 ({x0:.3f}, {y0:.3f}, {z0:.3f}) — 이 자세(대기위치)에서 시작한다")
    plan = [("기준", 0, 0, 0, 0), ("+y", 0, STEP, 0, 0), ("-y", 0, -STEP, 0, 0), ("+x", STEP, 0, 0, 0),
            ("-x", -STEP, 0, 0, 0), ("들기", 0, 0, LIFT, 0), ("손목+", 0, 0, 0, ROLL), ("손목-", 0, 0, 0, -ROLL),
            ("기준", 0, 0, 0, 0)]
    obs = []
    try:
        for name, dx, dy, dz, dr in plan:
            q = cal.ik_exact(x0 + dx, y0 + dy, z0 + dz, yaw0 + dr)
            node.move_joints(q, f"보정 {name}")
            Tw, Rw = cal.tool_pose()
            img = cal.frame()
            b = cal.blob(img)
            print(f"  {name:4s} 손끝 ({Tw[0]:.3f}, {Tw[1]:.3f}, {Tw[2]:.3f}) → " +
                  (f"공 영상 ({b[0]:.1f}, {b[1]:.1f}) 지름 {b[2]:.1f} px" if b else "공 못 찾음"))
            if b:
                obs.append((Tw, Rw, np.array(b)))
                cv2.imwrite(os.path.expanduser(f"~/omx_calib_{len(obs)}.jpg"), img)
    except StepFailed as e:
        print(f"❌ {e}")
    finally:
        ex.shutdown()
    if len(obs) < 6:
        raise SystemExit("관측이 모자란다(6개 이상 필요) — 공을 화면 가운데 근처에 두고 다시")

    H, W = 480, 640
    # 초기값: 화각 90° · 카메라는 손끝 뒤 3 cm · 공은 기준 영상 방향으로
    f0 = (W / 2) / math.tan(math.radians(45))
    T, R, m = obs[0]
    guess = [f0, 0.0, 0.0, -0.03, 0.0, 0.0, 0.0, T[0], T[1], 0.06]

    def resid(p):
        return np.concatenate([(project(p, T, R, W, H) - m) * [1, 1, 0.5] for T, R, m in obs])

    best = None
    for rz0 in np.radians([0, 90, 180, 270]):            # 영상이 손끝에 대해 몇 도 돌아 붙었는지 모른다 → 네 방향에서 출발
        g = list(guess)
        g[4] = rz0
        r = least_squares(resid, g, bounds=([100, -0.1, -0.1, -0.15, -10, -0.5, -0.5, -1, -1, 0.02],
                                            [1500, 0.1, 0.1, 0.05, 10, 0.5, 0.5, 1, 1, 0.15]))
        if best is None or r.cost < best.cost:
            best = r
    p = best.x
    rms = math.sqrt(np.mean(resid(p) ** 2))
    f, cx_, cy_, cz_, rz, rx, ry, X, Y, D = p
    hfov = 2 * math.degrees(math.atan((W / 2) / f))
    print(f"\n━━━ 결과 (잔차 RMS {rms:.1f} px · 관측 {len(obs)}개) ━━━")
    print(f"초점거리 f = {f:.0f} px  → 가로 화각 {hfov:.1f}°")
    print(f"카메라 위치(손끝 기준, m): x {cx_:+.4f} (닫힘 방향) · y {cy_:+.4f} · z {cz_:+.4f} (− = 손끝보다 뒤)")
    print(f"카메라 방향: 손끝 축 둘레 {math.degrees(rz):.1f}° · 기울기 {math.degrees(rx):.1f}°, {math.degrees(ry):.1f}°")
    print(f"공: 위치 ({X:.3f}, {Y:.3f}) · 지름 {D * 100:.1f} cm")

    # scene_omx.yaml camera_mount(부모 link5 · 카메라 몸체 x = 보는 방향) 로 환산
    #   link5 → tool_tip: 위치 (tip_open, -0.00165, 0) · 회전 rpy(90°, 0, 90°)
    R_5t = rot("z", math.pi / 2) @ rot("x", math.pi / 2)
    t_5t = np.array([float(scene["gripper"]["tip_open"]), -0.00165, 0.0])
    R_tc = rot("z", rz) @ rot("x", rx) @ rot("y", ry)
    R_ob = (rot("z", -math.pi / 2) @ rot("x", -math.pi / 2)).T     # 몸체 → 광학 = rpy(-90°, 0, -90°) 의 역
    R_5b = R_5t @ R_tc @ R_ob
    t_5b = t_5t + R_5t @ np.array([cx_, cy_, cz_])
    roll = math.atan2(R_5b[2, 1], R_5b[2, 2])
    pitch = math.asin(-max(-1.0, min(1.0, R_5b[2, 0])))
    yaw = math.atan2(R_5b[1, 0], R_5b[0, 0])
    out = {"camera_mount": {"parent": "link5", "xyz": [round(float(v), 4) for v in t_5b],
                            "rpy_deg": [round(math.degrees(a), 1) for a in (roll, pitch, yaw)]},
           "hfov_deg": round(hfov, 1)}
    print("\nscene_omx.yaml 에 넣을 값:\n" + yaml.safe_dump(out, allow_unicode=True, sort_keys=False))
    if rms > 5:
        print("⚠ 잔차가 크다 — 공이 화면 가장자리였거나(광각 왜곡) 움직였을 수 있다. 공을 가운데 두고 다시.")
    rclpy.shutdown()


if __name__ == "__main__":
    main()
