#!/usr/bin/env python3
"""물체 인식 노드 (USB 카메라 판) — 깊이 없는 컬러 영상에서 테이블 위 색깔 물체를 찾는다.

■ 아이디어
  깊이 카메라(D405)는 점마다 거리를 주지만, 일반 웹캠은 '어느 방향에 보이는가'만 안다.
  ⇒ 물체가 테이블 위에 놓여 있고 높이를 안다면, 그 방향의 광선이 '테이블 + 높이' 평면과
     만나는 곳이 물체 윗면이다.

  ① 색(HSV)으로 물체 픽셀 고르기 — 색마다 물체 높이를 scene_omx.yaml 에 정해 둔다
  ② 덩어리(윤곽선)마다 픽셀 → 카메라 광선 (카메라 내부 파라미터: camera_info 또는 화각)
  ③ 광선을 world 로 돌려(TF) 평면 z = 테이블 + 높이 와 만나는 점 = 윗면 위의 점
  ④ 윗면 점들에 최소 외접 사각형 → 모양(round/box)·폭·방향 (깊이판 object_detector 와 같은 방법)

■ 한계 (알고 쓰기)
  · 높이는 재지 않는다 — 색별 설정값. 높이가 틀리면 위치가 카메라 쪽(또는 반대쪽)으로 밀린다.
  · 카메라가 비스듬하면 옆면도 같은 색으로 보여 폭이 조금 크게 나온다.
  · 손에 든 물체가 그대로 남아 있는지(손가락 사이 확인)는 보지 않는다.

■ 토픽
  입력  /camera1/image_raw, /camera1/camera_info
  출력  /edu_pick/objects · /edu_pick/objects_markers · /edu_pick/debug_image · /edu_pick/detector_timing
"""
import json
import math
import time

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import CameraInfo, Image, JointState
from std_msgs.msg import String
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import MarkerArray

from edu_pick_msgs.msg import DetectedObject, DetectedObjectArray
from edu_pick_perception.common import default_scene_file, object_markers, quat_to_mat
from edu_pick_perception.object_detector import ObjectDetector


class ColorDetector(Node):
    def __init__(self):
        super().__init__("object_detector")     # 같은 이름 — pick_place 는 어느 인식 노드인지 몰라도 된다
        self.declare_parameter("image_topic", "/camera1/image_raw")
        self.declare_parameter("info_topic", "/camera1/camera_info")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("scene_file", default_scene_file())
        self.declare_parameter("rate", 3.0)
        self.world = self.get_parameter("world_frame").value
        with open(self.get_parameter("scene_file").value) as f:
            scene = yaml.safe_load(f)
        self.cfg = scene["detection"]
        self.cam_cfg = scene.get("camera", {})
        self.table_z = float(scene["table"]["top_z"])
        self.max_open = float(scene["gripper"]["max_open"])
        self.colors = self.cfg["colors"]

        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self._last, self._K, self._D, self._warned_k = None, None, None, False
        # 손목 카메라 화면에서 그리퍼 턱이 차지하는 영역(정규화 [u, v] 다각형) — 그 안의 색은 버리고,
        # 그 영역에 닿은 덩어리는 '가려짐'(일부만 보임 → 크기·위치가 틀림)으로 표시한다
        self._occ_poly = self.cam_cfg.get("occlusion_polygon") or []
        self._occ = None                          # (W, H, 영역, 닿음 판정용 영역)
        # 턱을 이 각도[rad] 이상 벌리면 턱이 화면 밖으로 나간다 → 가림 영역을 쓰지 않는다
        #   (OMX-F 실기 11-11 촬영: 0.8 rad 에서 턱이 화면에 전혀 없음 · 닫으면 화면의 32 %)
        self._occ_clear = self.cam_cfg.get("occlusion_clear_above")
        self._grip_joint = scene["gripper"].get("joint")
        self._grip_pos = None
        if self._occ_clear is not None:
            self.create_subscription(JointState, "/joint_states", self._on_js, 10)
        self.create_subscription(Image, self.get_parameter("image_topic").value,
                                 lambda m: setattr(self, "_last", m), 2)
        self.create_subscription(CameraInfo, self.get_parameter("info_topic").value, self._on_info, 2)
        self.pub = self.create_publisher(DetectedObjectArray, "/edu_pick/objects", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/edu_pick/objects_markers", 10)
        self.pub_img = self.create_publisher(Image, "/edu_pick/debug_image", 2)
        self.pub_timing = self.create_publisher(String, "/edu_pick/detector_timing", 10)
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.process)
        self.get_logger().info(f"물체 인식(USB 카메라·색) 시작 — 입력 {self.get_parameter('image_topic').value} · "
                               f"색 {[c['name'] for c in self.colors]}")

    def _on_js(self, msg):
        if self._grip_joint in msg.name:
            self._grip_pos = msg.position[msg.name.index(self._grip_joint)]

    def _on_info(self, msg):
        K = np.array(msg.k, float).reshape(3, 3)
        self._K = K if K[0, 0] > 0 else None       # 보정 안 된 카메라는 0 으로 온다
        # 렌즈 왜곡 계수 — 광각 렌즈(OMX 키트 카메라)는 가장자리가 휘어 보인다. 보정 파일이 있으면 펴서 쓴다
        self._D = np.array(msg.d, float) if self._K is not None and len(msg.d) and np.any(msg.d) else None

    def intrinsics(self, w, h):
        """카메라 내부 파라미터(fx, fy, cx, cy). 보정값이 없으면 화각(hfov_deg)으로 근사한다."""
        if self._K is not None:
            return self._K[0, 0], self._K[1, 1], self._K[0, 2], self._K[1, 2]
        if not self._warned_k:
            self.get_logger().warn(f"camera_info 보정값 없음 — 가로 화각 {self.cam_cfg.get('hfov_deg', 60)}° 로 근사합니다 "
                                   "(위치 오차가 크면 카메라 보정: README 참고)")
            self._warned_k = True
        f = (w / 2) / math.tan(math.radians(float(self.cam_cfg.get("hfov_deg", 60.0)) / 2))
        return f, f, w / 2, h / 2

    # ------------------------------------------------------------------ 처리
    def process(self):
        msg, self._last = self._last, None
        if msg is None:
            return
        t0 = time.perf_counter()
        try:
            t = self.tf.lookup_transform(self.world, msg.header.frame_id, Time())
        except Exception as e:
            self.get_logger().warn(f"TF 없음 {self.world}←{msg.header.frame_id}: {e}", throttle_duration_sec=5.0)
            return
        bgr = self.bridge.imgmsg_to_cv2(msg, "bgr8")
        H, W = bgr.shape[:2]
        fx, fy, cx, cy = self.intrinsics(W, H)
        R = quat_to_mat(t.transform.rotation)
        cam = np.array([t.transform.translation.x, t.transform.translation.y, t.transform.translation.z])
        hsv = cv2.cvtColor(cv2.GaussianBlur(bgr, (5, 5), 0), cv2.COLOR_BGR2HSV)
        k5 = np.ones((5, 5), np.uint8)
        c = self.cfg
        occ, occ_near = self.occlusion(W, H)

        found, all_mask = [], np.zeros((H, W), np.uint8)
        for col in self.colors:
            mask = cv2.inRange(hsv, np.array(col["hsv"][0]), np.array(col["hsv"][1]))
            if "hsv2" in col:                       # 빨강은 색상환 양 끝(0 근처·179 근처)에 걸친다
                mask |= cv2.inRange(hsv, np.array(col["hsv2"][0]), np.array(col["hsv2"][1]))
            if occ is not None:                     # 턱 구멍 사이로 비친 색 등 — 그리퍼 영역 안은 버린다
                mask[occ > 0] = 0
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            all_mask |= mask
            sphere = str(col["height"]) == "sphere"
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 30:
                    continue
                blob = np.zeros((H, W), np.uint8)
                cv2.drawContours(blob, [cnt], -1, 255, -1)
                vs, us = np.nonzero(blob)
                o = self.locate(us, vs, R, cam, fx, fy, cx, cy, col, sphere)
                if o is None:
                    continue
                # 인식 영역(roi) 밖 · 영상 가장자리에 걸린 덩어리
                if not (c["roi_x"][0] < o.top_center.x < c["roi_x"][1] and c["roi_y"][0] < o.top_center.y < c["roi_y"][1]):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                # 잘림·가려짐은 다른 판정(거리 초과 등)보다 먼저 표시한다 — 일부만 보인 덩어리는 위치·크기 자체가 틀려서
                #   pick_place 가 두 자세 결과를 합칠 때 '온전히 보인 쪽'을 고르는 기준이 된다 (11-11 가상: 잘린 뒷줄이 거리 초과로만 표시돼 온전한 것으로 취급됐다)
                if x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2:
                    o.graspable, o.note = False, "가장자리에 잘림"
                elif occ_near is not None and np.any(occ_near[blob > 0]):
                    o.graspable, o.note = False, "그리퍼에 가려짐"
                found.append((o, cnt))
        for i, (o, _) in enumerate(found):
            o.id = f"obj_{i}"
        if occ is not None:                         # 화면에 그리퍼 영역 테두리
            cv2.polylines(bgr, [self._occ[4]], True, (255, 0, 255), 1)
        self._publish(found, msg.header.stamp, bgr, all_mask)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pub_timing.publish(String(data=json.dumps(
            {"stamp": stamp, "objects": len(found), "ms": {"합계": round((time.perf_counter() - t0) * 1000, 2)}},
            ensure_ascii=False)))

    def occlusion(self, W, H):
        """그리퍼 영역 마스크(영역, 영역+여유 2 px). 설정이 없으면 (None, None)."""
        if len(self._occ_poly) < 3:
            return None, None
        if self._occ_clear is not None and self._grip_pos is not None and self._grip_pos >= float(self._occ_clear):
            return None, None                     # 턱을 활짝 벌려 화면에 없다
        if self._occ is None or self._occ[:2] != (W, H):
            pts = np.array([[round(u * (W - 1)), round(v * (H - 1))] for u, v in self._occ_poly], np.int32)
            m = np.zeros((H, W), np.uint8)
            cv2.fillPoly(m, [pts], 255)
            # 닿음 판정 여유 2 px — 6 px 였을 때 온전히 보인 토마토(round 5.0 cm)를 '가려짐'으로 버렸다(11-11 실기).
            #   다각형 자체가 실제 턱보다 약 6 px 위로 그려져 있다
            self._occ = (W, H, m, cv2.dilate(m, np.ones((5, 5), np.uint8)), pts.reshape(-1, 1, 2))
        return self._occ[2], self._occ[3]

    def locate(self, us, vs, R, cam, fx, fy, cx, cy, col, sphere):
        """덩어리 픽셀 → 광선 → 평면과의 교점 → DetectedObject.
        평면 높이: 색마다 정한 물체 높이(윗면) · sphere 면 지름의 절반(위에서 보이는 윤곽 = 적도)"""
        # ② 픽셀 → 광학 프레임 광선
        if self._D is not None:                   # 왜곡 보정: 휘어 보인 픽셀을 바늘구멍 카메라 좌표로
            n = cv2.undistortPoints(np.stack([us, vs], -1).astype(np.float32).reshape(-1, 1, 2),
                                    self._K, self._D).reshape(-1, 2)
            rays = np.stack([n[:, 0], n[:, 1], np.ones(len(n))], -1) @ R.T
        else:
            rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1) @ R.T
        height = 0.05 if sphere else float(col["height"])
        o = None
        for _ in range(3 if sphere else 1):       # 공: 지름 → 높이 → 다시 지름 (2~3 번이면 수렴)
            plane_z = self.table_z + (height / 2 if sphere else height)
            # ③ world 광선과 평면의 교점
            with np.errstate(divide="ignore", invalid="ignore"):
                s = (plane_z - cam[2]) / rays[:, 2]
            ok = np.isfinite(s) & (s > 0)
            if ok.sum() < 20:
                return None
            xy = cam[:2] + s[ok, None] * rays[ok, :2]
            o = self.describe(xy, self.table_z + height, height, col["name"])
            if o is None or not sphere:
                return o
            height = o.width
        o.top_center.z, o.height = self.table_z + o.width, o.width
        return o

    def describe(self, xy, top_z, height, color):
        """윗면 위의 점(xy, world) → DetectedObject. 모양·폭·방향은 깊이판과 같은 규칙."""
        c = self.cfg
        (cx, cy), width, length, short_dir, area = ObjectDetector.robust_box(xy)
        if area < float(c["min_area"]) or width < float(c.get("min_width", 0.0)):
            return None
        fill = area / max(width * length, 1e-9)
        o = DetectedObject()
        if length / max(width, 1e-9) < 1.2 and fill < 0.88:
            o.shape, o.yaw = "round", float("nan")
            width = length = (width + length) / 2
        else:
            o.shape = "box"
            yaw = math.atan2(float(short_dir[1]), float(short_dir[0]))
            while yaw > math.pi / 2:
                yaw -= math.pi
            while yaw <= -math.pi / 2:
                yaw += math.pi
            o.yaw = yaw
        o.top_center = Point(x=float(cx), y=float(cy), z=float(top_z))
        o.width, o.length, o.height = width, max(length, width), height
        o.graspable = width <= float(c["max_grip_width"])
        o.note = "" if o.graspable else f"너무 넓음({color} · {width * 100:.1f} cm > {float(c['max_grip_width']) * 100:.1f} cm)"
        # 손이 닿는 거리: 로봇 중심(world 원점)에서 max_reach 를 넘으면 위에서 수직으로 집을 수 없다
        r = math.hypot(cx, cy)
        if o.graspable and r > float(c.get("max_reach", 1e9)):
            o.graspable, o.note = False, f"손이 닿지 않음(로봇에서 {r * 100:.0f} cm > {float(c['max_reach']) * 100:.0f} cm)"
        return o

    def _publish(self, objects, stamp, bgr, mask):
        out = DetectedObjectArray()
        out.header.stamp = stamp
        out.header.frame_id = self.world
        out.objects = [o for o, _ in objects]
        self.pub.publish(out)
        self.pub_mk.publish(object_markers(out.objects, self.world))
        if self.pub_img.get_subscription_count() == 0:
            return
        img = bgr.copy()
        img[mask > 0] = (0.5 * img[mask > 0] + [0, 110, 0]).astype(np.uint8)
        for o, cnt in objects:
            colr = (0, 255, 0) if o.graspable else (0, 0, 255)
            cv2.drawContours(img, [cnt], -1, colr, 2)
            x, y, _, _ = cv2.boundingRect(cnt)
            cv2.putText(img, f"{o.id} {o.shape} {o.width * 100:.1f}cm" + ("" if o.graspable else " X"),
                        (x, max(15, y - 5)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, colr, 2)
        cv2.putText(img, f"objects = {len(objects)}", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))


def main():
    rclpy.init()
    node = ColorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        if rclpy.ok():
            rclpy.shutdown()


if __name__ == "__main__":
    main()
