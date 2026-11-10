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
from sensor_msgs.msg import CameraInfo, Image
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

        found, all_mask = [], np.zeros((H, W), np.uint8)
        for col in self.colors:
            mask = cv2.inRange(hsv, np.array(col["hsv"][0]), np.array(col["hsv"][1]))
            if "hsv2" in col:                       # 빨강은 색상환 양 끝(0 근처·179 근처)에 걸친다
                mask |= cv2.inRange(hsv, np.array(col["hsv2"][0]), np.array(col["hsv2"][1]))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)
            all_mask |= mask
            plane_z = self.table_z + float(col["height"])
            contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            for cnt in contours:
                if cv2.contourArea(cnt) < 30:
                    continue
                blob = np.zeros((H, W), np.uint8)
                cv2.drawContours(blob, [cnt], -1, 255, -1)
                vs, us = np.nonzero(blob)
                # ② 픽셀 → 광학 프레임 광선 → ③ world 광선과 윗면 평면의 교점
                if self._D is not None:           # 왜곡 보정: 휘어 보인 픽셀을 바늘구멍 카메라 좌표로
                    n = cv2.undistortPoints(np.stack([us, vs], -1).astype(np.float32).reshape(-1, 1, 2),
                                            self._K, self._D).reshape(-1, 2)
                    rays = np.stack([n[:, 0], n[:, 1], np.ones(len(n))], -1) @ R.T
                else:
                    rays = np.stack([(us - cx) / fx, (vs - cy) / fy, np.ones_like(us, float)], -1) @ R.T
                with np.errstate(divide="ignore", invalid="ignore"):
                    s = (plane_z - cam[2]) / rays[:, 2]
                ok = np.isfinite(s) & (s > 0)
                if ok.sum() < 20:
                    continue
                xy = cam[:2] + s[ok, None] * rays[ok, :2]
                o = self.describe(xy, plane_z, float(col["height"]), col["name"])
                if o is None:
                    continue
                # 인식 영역(roi) 밖 · 영상 가장자리에 걸린 덩어리
                if not (c["roi_x"][0] < o.top_center.x < c["roi_x"][1] and c["roi_y"][0] < o.top_center.y < c["roi_y"][1]):
                    continue
                x, y, w, h = cv2.boundingRect(cnt)
                if (x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2) and o.graspable:
                    o.graspable, o.note = False, "가장자리에 잘림"
                found.append((o, cnt))
        for i, (o, _) in enumerate(found):
            o.id = f"obj_{i}"
        self._publish(found, msg.header.stamp, bgr, all_mask)
        stamp = msg.header.stamp.sec + msg.header.stamp.nanosec * 1e-9
        self.pub_timing.publish(String(data=json.dumps(
            {"stamp": stamp, "objects": len(found), "ms": {"합계": round((time.perf_counter() - t0) * 1000, 2)}},
            ensure_ascii=False)))

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
