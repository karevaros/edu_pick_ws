#!/usr/bin/env python3
"""OMX 손목 카메라 보정 — 바닥에 놓은 ChArUco 판(tools/make_charuco_board.py)을 여러 팔 자세에서 찍어
렌즈(초점거리·중심·왜곡) + 카메라 장착 위치(link5 기준) + 판 위치(world)를 한 번에 맞춘다.

    python3 tools/omx_charuco_calib.py capture   # 실기: 팔을 자세 목록대로 옮기며 찍음 (bringup 이 떠 있어야 함)
    python3 tools/omx_charuco_calib.py solve     # 찍은 것으로 계산 → scene_omx.yaml 에 넣을 값 출력

판을 어디 놓았는지 정확히 몰라도 된다 — 판 위치(x, y, 방향)도 함께 추정한다(놓은 위치는 초기값일 뿐).
판 높이는 테이블 윗면(scene table.top_z)으로 고정한다.
그리퍼 턱에 가린 영역(scene camera.occlusion_polygon) 근처의 모서리는 버린다.
"""
import json
import math
import os
import sys
import time

import cv2
import numpy as np
import yaml

OUT = os.path.expanduser("~/.ros/omx_charuco_calib")
SCENE = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "src", "edu_pick_description", "config", "scene_omx.yaml")
COLS, ROWS, SQUARE, MARKER = 9, 6, 0.025, 0.018
# 판 놓은 자리(초기값): 로봇 쪽 긴 변 가운데가 정면 0.10 m · 좌우 가운데
BOARD_NEAR_X = 0.10
# 찍을 팔 자세 [deg] — 모두 손끝이 높이 떠 있는 자세 (capture 가 FK 로 다시 확인)
POSES = [
    [0, -20, -20, 92, 0], [0, -35, 5, 100, 0], [0, -27, -8, 96, 0],
    [12, -20, -20, 92, 0], [-12, -20, -20, 92, 0],
    [12, -35, 5, 100, 0], [-12, -35, 5, 100, 0],
    [0, -27, -8, 96, 25], [0, -27, -8, 96, -25],
]
MIN_TIP_Z = 0.12


def board():
    return cv2.aruco.CharucoBoard((COLS, ROWS), SQUARE, MARKER, cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50))


def scene():
    with open(SCENE) as f:
        return yaml.safe_load(f)


# ------------------------------------------------------------------ capture
def capture():
    import rclpy
    from rclpy.action import ActionClient
    from rclpy.duration import Duration
    from rclpy.node import Node
    from rclpy.time import Time
    from control_msgs.action import FollowJointTrajectory
    from moveit_msgs.srv import GetPositionFK
    from sensor_msgs.msg import Image, JointState
    from trajectory_msgs.msg import JointTrajectoryPoint
    import tf2_ros

    os.makedirs(OUT, exist_ok=True)
    rclpy.init()
    n = Node("omx_charuco_calib")
    js, img = {}, {}
    n.create_subscription(JointState, "/joint_states", lambda m: js.update(zip(m.name, m.position)), 10)

    def on_img(m):
        a = np.frombuffer(m.data, np.uint8).reshape(m.height, m.width, -1)
        img["a"] = cv2.cvtColor(a, cv2.COLOR_RGB2BGR) if m.encoding == "rgb8" else a.copy()
        img["t"] = time.time()
    n.create_subscription(Image, "/camera1/image_raw", on_img, 2)
    buf = tf2_ros.Buffer()
    tf2_ros.TransformListener(buf, n)
    fk = n.create_client(GetPositionFK, "/compute_fk")
    ac = ActionClient(n, FollowJointTrajectory, "/arm_controller/follow_joint_trajectory")
    names = [f"joint{i}" for i in range(1, 6)]

    def spin(sec):
        t = time.time()
        while time.time() - t < sec:
            rclpy.spin_once(n, timeout_sec=0.05)

    def tip_z(q):
        r = GetPositionFK.Request()
        r.header.frame_id, r.fk_link_names = "world", ["tool_tip", "link5"]
        r.robot_state.joint_state.name, r.robot_state.joint_state.position = names, [math.radians(v) for v in q]
        f = fk.call_async(r)
        rclpy.spin_until_future_complete(n, f, timeout_sec=5)
        return min(p.pose.position.z for p in f.result().pose_stamped)

    if not fk.wait_for_service(timeout_sec=10) or not ac.wait_for_server(timeout_sec=10):
        sys.exit("move_group(/compute_fk) 또는 arm_controller 가 없습니다 — bringup 을 먼저 띄우세요")
    spin(1.0)
    rec = []
    for k, q in enumerate(POSES):
        z = tip_z(q)
        if z < MIN_TIP_Z:
            print(f"[{k}] {q} 건너뜀 — 손끝 높이 {z:.3f} m < {MIN_TIP_Z}")
            continue
        g = FollowJointTrajectory.Goal()
        g.trajectory.joint_names = names
        p = JointTrajectoryPoint(positions=[math.radians(v) for v in q], velocities=[0.0] * 5,
                                 time_from_start=Duration(seconds=4.0).to_msg())
        g.trajectory.points = [p]
        f = ac.send_goal_async(g)
        rclpy.spin_until_future_complete(n, f, timeout_sec=10)
        h = f.result()
        if h is None or not h.accepted:
            print(f"[{k}] 목표 거부 — 중단")
            break
        r = h.get_result_async()
        rclpy.spin_until_future_complete(n, r, timeout_sec=15)
        code = r.result().result.error_code if r.done() else None
        spin(1.5)                                      # 흔들림 가라앉기 + 새 영상
        t_req = time.time()
        while img.get("t", 0) < t_req:
            spin(0.05)
        tf = buf.lookup_transform("world", "link5", Time()).transform
        cur = [math.degrees(js[j]) for j in names]
        fn = os.path.join(OUT, f"view_{k:02d}.png")
        cv2.imwrite(fn, img["a"])
        rec.append({"image": fn, "joints_deg": cur, "cmd_deg": q, "code": code,
                    "link5_t": [tf.translation.x, tf.translation.y, tf.translation.z],
                    "link5_q": [tf.rotation.x, tf.rotation.y, tf.rotation.z, tf.rotation.w]})
        n_c = detect(img["a"], scene())[0]
        print(f"[{k}] 목표 {q} → 실제 {[round(v, 1) for v in cur]} · 결과코드 {code} · 판 모서리 {len(n_c)}개")
    with open(os.path.join(OUT, "views.json"), "w") as f:
        json.dump(rec, f, indent=1)
    print(f"저장: {OUT}/views.json ({len(rec)}장) — 팔은 마지막 자세에 있습니다. go_pose home 으로 접으세요")


# ------------------------------------------------------------------ detect / solve
def detect(bgr, sc):
    """ChArUco 모서리 → (판 3D 점[N,3], 영상 점[N,2], ids). 그리퍼 영역 근처는 버림."""
    gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
    H, W = gray.shape
    b = board()
    # 기본값으로는 사진당 13개 남짓 — 광각 왜곡·흐림 때문에 마커를 많이 놓친다 → 조정(22~29개)
    dp = cv2.aruco.DetectorParameters()
    dp.adaptiveThreshWinSizeMax = 53
    dp.cornerRefinementMethod = cv2.aruco.CORNER_REFINE_SUBPIX
    dp.minMarkerPerimeterRate = 0.02
    chp = cv2.aruco.CharucoParameters()
    chp.minMarkers, chp.tryRefineMarkers = 1, True
    cc, ci, _, _ = cv2.aruco.CharucoDetector(b, chp, dp).detectBoard(gray)
    if ci is None or len(ci) == 0:
        return np.zeros((0, 3)), np.zeros((0, 2)), np.zeros(0, int)
    cc, ci = cc.reshape(-1, 2), ci.reshape(-1)
    poly = sc["camera"].get("occlusion_polygon") or []
    keep = np.ones(len(ci), bool)
    if len(poly) >= 3:
        pts = np.array([[u * (W - 1), v * (H - 1)] for u, v in poly], np.float32)
        for i, (u, v) in enumerate(cc):
            keep[i] = cv2.pointPolygonTest(pts, (float(u), float(v)), True) < -15.0   # 영역 밖 15 px 이상
    obj = b.getChessboardCorners()[ci[keep]]
    return obj, cc[keep], ci[keep]


def q2m(q):
    x, y, z, w = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def rpy_m(r, p, y):
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    Rx = np.array([[1, 0, 0], [0, cr, -sr], [0, sr, cr]])
    Ry = np.array([[cp, 0, sp], [0, 1, 0], [-sp, 0, cp]])
    Rz = np.array([[cy, -sy, 0], [sy, cy, 0], [0, 0, 1]])
    return Rz @ Ry @ Rx


# 카메라 몸체(x = 보는 방향) → 광학 프레임(z 앞 · x 오른쪽 · y 아래) : URDF usb_cam_optical_joint rpy (−π/2, 0, −π/2)
R_BODY_OPT = rpy_m(-math.pi / 2, 0, -math.pi / 2)


def board_to_world(p, ztab):
    """판 좌표 (bx: 인쇄면 오른쪽 · by: 인쇄면 아래 = 로봇 쪽) → world. p = (x0, y0, yaw) 판 원점(인쇄면 왼쪽 위)의 위치·방향."""
    x0, y0, yaw = p
    # yaw = 0 이면 bx → world −y, by → world −x (로봇 쪽 변이 로봇을 향하게 놓았을 때)
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s], [s, c]]) @ np.array([[0, -1], [-1, 0]])
    return lambda P: np.c_[(R @ P[:, :2].T).T + [x0, y0], np.full(len(P), ztab)]


def solve():
    from scipy.optimize import least_squares
    sc = scene()
    ztab = float(sc["table"]["top_z"])
    with open(os.path.join(OUT, "views.json")) as f:
        views = json.load(f)
    data = []
    for v in views:
        obj, uv, ids = detect(cv2.imread(v["image"]), sc)
        if len(uv) >= 15:     # 점이 적으면(특히 한 줄로 늘어서면) 렌즈 보정이 엉뚱한 답을 낸다 — 11-11: 8점 사진 1장이 전체 RMS 를 0.4 → 25 px 로
            data.append((obj, uv, q2m(v["link5_q"]), np.array(v["link5_t"]), v))
        print(f"{os.path.basename(v['image'])}: 모서리 {len(uv)}개 · 관절 {[round(a, 1) for a in v['joints_deg']]}")
    if len(data) < 3:
        sys.exit("모서리 15개 이상 보인 사진이 3장 미만 — 판 위치·조명을 확인하세요")
    W, H = 640, 480
    cm = sc["camera_mount"]
    hfov0 = math.radians(float(sc["camera"].get("hfov_deg", 47.0)))
    # 변수: f, cx, cy, k1, k2 | 카메라 장착 xyz, rpy | 판 x0, y0, yaw
    x0 = np.r_[(W / 2) / math.tan(hfov0 / 2), W / 2, H / 2, 0.0, 0.0,
               cm["xyz"], [math.radians(a) for a in cm["rpy_deg"]],
               BOARD_NEAR_X + ROWS * SQUARE, COLS * SQUARE / 2, 0.0]

    def project(x, obj, R5, t5):
        f, cx, cy, k1, k2 = x[:5]
        Rm = R5 @ rpy_m(*x[8:11]) @ R_BODY_OPT
        tm = t5 + R5 @ x[5:8]
        Pw = board_to_world(x[11:14], ztab)(obj)
        Pc = (Pw - tm) @ Rm                      # = Rmᵀ (Pw − tm)
        xn, yn = Pc[:, 0] / Pc[:, 2], Pc[:, 1] / Pc[:, 2]
        r2 = xn * xn + yn * yn
        d = 1 + k1 * r2 + k2 * r2 * r2
        return np.c_[f * xn * d + cx, f * yn * d + cy], Pc[:, 2]

    def resid(x):
        out = [(project(x, obj, R5, t5)[0] - uv).ravel() for obj, uv, R5, t5, _ in data]
        out.append([(x[1] - W / 2) / 40, (x[2] - H / 2) / 40])   # 중심은 화면 가운데 근처로 약하게 묶음
        return np.concatenate(out)

    # ① 렌즈: OpenCV 표준 보정(평면 판 여러 장) — 장착 위치와 무관하게 초점거리·중심·왜곡을 먼저 구한다
    K0 = np.array([[x0[0], 0, W / 2], [0, x0[0], H / 2], [0, 0, 1]])
    flags = (cv2.CALIB_USE_INTRINSIC_GUESS | cv2.CALIB_FIX_ASPECT_RATIO | cv2.CALIB_ZERO_TANGENT_DIST
             | cv2.CALIB_FIX_K3)
    rms, K, D, _, _ = cv2.calibrateCamera([o.astype(np.float32) for o, *_ in data],
                                          [u.astype(np.float32).reshape(-1, 1, 2) for _, u, *_ in data],
                                          (W, H), K0, None, flags=flags)
    print(f"  ① 렌즈 보정(calibrateCamera): RMS {rms:.2f} px · f {K[0, 0]:.1f} · 중심 ({K[0, 2]:.1f}, {K[1, 2]:.1f}) · k1 {D.ravel()[0]:.3f} k2 {D.ravel()[1]:.3f}")
    x0[:5] = [K[0, 0], K[0, 2], K[1, 2], D.ravel()[0], D.ravel()[1]]
    # ② 렌즈 고정 → 장착 위치·판 위치
    fixed = x0[:5].copy()
    s2 = least_squares(lambda y: resid(np.r_[fixed, y]), x0[5:], loss="huber", f_scale=2.0)
    print(f"  ② 렌즈 고정 후 장착·판: 재투영 중앙 {np.median(np.abs(s2.fun)):.2f} px(축별)")
    # ③ 전체 다듬기
    sol = least_squares(resid, np.r_[fixed, s2.x], loss="huber", f_scale=2.0)
    x = sol.x
    per = []
    for obj, uv, R5, t5, v in data:
        e = np.linalg.norm(project(x, obj, R5, t5)[0] - uv, axis=1)
        per.append((os.path.basename(v["image"]), len(e), float(np.median(e)), float(e.max())))
    allerr = np.concatenate([np.linalg.norm(project(x, o, R, t)[0] - u, axis=1) for o, u, R, t, _ in data])
    f, cx, cy, k1, k2 = x[:5]
    hfov = math.degrees(2 * math.atan((W / 2) / f))
    print("\n── 결과 ──────────────────────────────")
    for nme, nn, med, mx in per:
        print(f"  {nme}: 점 {nn} · 재투영 오차 중앙 {med:.2f} px · 최대 {mx:.2f} px")
    print(f"  전체 재투영 오차: 중앙 {np.median(allerr):.2f} px · 95% {np.percentile(allerr, 95):.2f} px · 점 {len(allerr)}")
    print(f"  렌즈: f {f:.1f} px (가로 화각 {hfov:.1f}°) · 중심 ({cx:.1f}, {cy:.1f}) · 왜곡 k1 {k1:.3f} k2 {k2:.3f}")
    print(f"  카메라 장착(link5): xyz [{x[5]:.4f}, {x[6]:.4f}, {x[7]:.4f}] · rpy_deg "
          f"[{math.degrees(x[8]):.1f}, {math.degrees(x[9]):.1f}, {math.degrees(x[10]):.1f}]")
    print(f"     (지금 scene: xyz {cm['xyz']} · rpy_deg {cm['rpy_deg']})")
    near_c = np.array([[COLS * SQUARE / 2, ROWS * SQUARE, 0.0]])          # 판 로봇 쪽 변 가운데
    nc = board_to_world(x[11:14], ztab)(near_c)[0]
    print(f"  판 위치 추정: 로봇 쪽 변 가운데 = 정면 {nc[0] * 100:.1f} cm · 좌우 {nc[1] * 100:+.1f} cm · 방향 {math.degrees(x[13]):+.1f}°"
          f"  (놓기로 한 자리: 정면 {BOARD_NEAR_X * 100:.0f} cm · 0)")
    f, cx, cy, k1, k2, hfov = (float(v) for v in (f, cx, cy, k1, k2, hfov))
    res = {"f": f, "cx": cx, "cy": cy, "k1": k1, "k2": k2, "hfov_deg": hfov,
           "camera_mount": {"parent": "link5", "xyz": [round(float(v), 4) for v in x[5:8]],
                            "rpy_deg": [round(math.degrees(float(v)), 2) for v in x[8:11]]},
           "board_near_center_xy": [float(nc[0]), float(nc[1])], "board_yaw_deg": math.degrees(x[13]),
           "reproj_median_px": float(np.median(allerr)), "reproj_p95_px": float(np.percentile(allerr, 95)),
           "points": int(len(allerr)), "views": len(data)}
    with open(os.path.join(OUT, "result.json"), "w") as fo:
        json.dump(res, fo, indent=1)
    # usb_cam 이 읽는 camera_info 파일(왜곡 포함)
    ci = {"image_width": W, "image_height": H, "camera_name": "omx_cam",
          "camera_matrix": {"rows": 3, "cols": 3, "data": [f, 0, cx, 0, f, cy, 0, 0, 1]},
          "distortion_model": "plumb_bob",
          "distortion_coefficients": {"rows": 1, "cols": 5, "data": [k1, k2, 0, 0, 0]},
          "rectification_matrix": {"rows": 3, "cols": 3, "data": [1, 0, 0, 0, 1, 0, 0, 0, 1]},
          "projection_matrix": {"rows": 3, "cols": 4, "data": [f, 0, cx, 0, 0, f, cy, 0, 0, 0, 1, 0]}}
    with open(os.path.join(OUT, "omx_cam.yaml"), "w") as fo:
        yaml.safe_dump(ci, fo, sort_keys=False)
    print(f"  저장: {OUT}/result.json · {OUT}/omx_cam.yaml (camera_info)")


if __name__ == "__main__":
    {"capture": capture, "solve": solve}[sys.argv[1] if len(sys.argv) > 1 else "solve"]()
