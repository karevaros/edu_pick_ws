#!/usr/bin/env python3
"""물체 인식 노드 — 위에서 내려다본 깊이 영상에서 테이블 위 물체를 찾는다.

■ 아이디어 (딥러닝 없이)
  카메라가 테이블을 수직으로 내려다보면, 테이블은 '거의 같은 높이의 평면'이다.
  ⇒ 테이블보다 높은 점들의 덩어리 = 물체.

  ① 점군(카메라 좌표) → TF 로 world(로봇 베이스) 좌표로 변환
  ② 테이블 높이 추정: 높이 히스토그램에서 가장 많은 높이 = 테이블 면
  ③ 테이블보다 min_height 이상 높은 점 = 물체 후보 (마스크)
  ④ 마스크에서 덩어리(윤곽선) 찾기 → 덩어리마다 3D 점을 모아
  ⑤ 윗면 점들의 외곽(xy)에 '최소 외접 사각형'을 씌운다
       · 가로세로가 비슷하고 사각형을 덜 채우면(원은 78.5%) → round (캔·병·공)
       · 아니면 → box
       · 짧은 변 길이 = 집을 폭, 짧은 변 방향 = 그리퍼가 닫힐 방향(yaw)
  ⑥ 결과 발행

■ 토픽
  입력  /camera/d405/depth/color/points      (sensor_msgs/PointCloud2, 격자형)
  출력  /edu_pick/objects                     (edu_pick_msgs/DetectedObjectArray)
        /edu_pick/objects_markers             (RViz 표시)
        /edu_pick/debug_image                 (단계별 결과 그림 — rqt_image_view 로 보기)
"""
import math

import cv2
import numpy as np
import rclpy
import yaml
from cv_bridge import CvBridge
from geometry_msgs.msg import Point
from rclpy.node import Node
from rclpy.time import Time
from sensor_msgs.msg import Image, PointCloud2
from tf2_ros import Buffer, TransformListener
from visualization_msgs.msg import Marker, MarkerArray

from edu_pick_msgs.msg import DetectedObject, DetectedObjectArray
from edu_pick_perception.common import default_scene_file, quat_to_mat, object_markers


def cloud_to_arrays(msg):
    """PointCloud2(격자형) → xyz (H,W,3) float32, rgb (H,W,3) uint8."""
    fields = {f.name: f.offset for f in msg.fields}
    dtype = np.dtype({"names": ["x", "y", "z", "rgb"],
                      "formats": ["<f4", "<f4", "<f4", "<u4"],
                      "offsets": [fields["x"], fields["y"], fields["z"], fields["rgb"]],
                      "itemsize": msg.point_step})
    a = np.frombuffer(bytes(msg.data), dtype=dtype, count=msg.width * msg.height)
    h, w = msg.height, msg.width
    xyz = np.stack([a["x"], a["y"], a["z"]], axis=-1).reshape(h, w, 3)
    c = a["rgb"].reshape(h, w)
    rgb = np.stack([(c >> 16) & 255, (c >> 8) & 255, c & 255], axis=-1).astype(np.uint8)
    return xyz, rgb


class ObjectDetector(Node):
    def __init__(self):
        super().__init__("object_detector")
        self.declare_parameter("cloud_topic", "/camera/d405/depth/color/points")
        self.declare_parameter("world_frame", "world")
        self.declare_parameter("scene_file", default_scene_file())
        self.declare_parameter("rate", 3.0)     # 초당 처리 횟수

        self.world = self.get_parameter("world_frame").value
        with open(self.get_parameter("scene_file").value) as f:
            scene = yaml.safe_load(f)
        self.cfg = scene["detection"]
        self.table_z_cfg = float(scene["table"]["top_z"])

        self.tf = Buffer()
        TransformListener(self.tf, self)
        self.bridge = CvBridge()
        self._last = None
        self.create_subscription(PointCloud2, self.get_parameter("cloud_topic").value,
                                 lambda m: setattr(self, "_last", m), 2)
        self.pub = self.create_publisher(DetectedObjectArray, "/edu_pick/objects", 10)
        self.pub_mk = self.create_publisher(MarkerArray, "/edu_pick/objects_markers", 10)
        self.pub_img = self.create_publisher(Image, "/edu_pick/debug_image", 2)
        self.create_timer(1.0 / float(self.get_parameter("rate").value), self.process)
        self.get_logger().info(f"물체 인식 시작 — 입력 {self.get_parameter('cloud_topic').value}")

    # ------------------------------------------------------------------ 처리
    def process(self):
        msg, self._last = self._last, None
        if msg is None:
            return
        if msg.height <= 1:
            self.get_logger().error("점군이 격자형이 아닙니다 — 카메라를 pointcloud.ordered_pc:=true 로 띄우세요.",
                                    throttle_duration_sec=5.0)
            return
        # ① 카메라 → world 변환. **점군을 찍은 시각**의 TF 를 쓴다
        #    (가장 최근 TF 를 쓰면 팔이 움직이는 중에 찍힌 점군이 엉뚱한 곳에 놓인다)
        try:
            t = self.tf.lookup_transform(self.world, msg.header.frame_id, Time.from_msg(msg.header.stamp))
        except Exception as e:
            self.get_logger().warn(f"TF 없음 {self.world}←{msg.header.frame_id}: {e}", throttle_duration_sec=5.0)
            return
        R = quat_to_mat(t.transform.rotation)
        p = t.transform.translation
        xyz_cam, rgb = cloud_to_arrays(msg)
        H, W, _ = xyz_cam.shape
        valid = np.isfinite(xyz_cam[..., 2]) & (xyz_cam[..., 2] > 0.05)
        P = xyz_cam.reshape(-1, 3) @ R.T + np.array([p.x, p.y, p.z])
        P = P.reshape(H, W, 3)
        X, Y, Z = P[..., 0], P[..., 1], P[..., 2]

        c = self.cfg
        roi = valid & (X > c["roi_x"][0]) & (X < c["roi_x"][1]) & (Y > c["roi_y"][0]) & (Y < c["roi_y"][1])
        if roi.sum() < 500:
            self._publish([], msg.header.stamp, rgb, None)
            return

        # ② 테이블 높이: ROI 안 높이 히스토그램의 최빈값(4 mm 간격)
        table_z = self.table_z_cfg
        if c.get("auto_table", True):
            zs = Z[roi]
            zs = zs[np.abs(zs - self.table_z_cfg) < 0.10]
            if zs.size > 500:
                hist, edges = np.histogram(zs, bins=np.arange(self.table_z_cfg - 0.10, self.table_z_cfg + 0.10, 0.004))
                k = int(np.argmax(hist))
                if hist[k] > 0.15 * zs.size:
                    table_z = float(0.5 * (edges[k] + edges[k + 1]))

        # ③ 물체 마스크
        hgt = Z - table_z
        mask = (roi & (hgt > c["min_height"]) & (hgt < c["max_height"])).astype(np.uint8) * 255
        k5 = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, k5)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, k5)

        # ④ 덩어리마다 ⑤ 모양·크기·방향
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        objects = []
        for cnt in contours:
            if cv2.contourArea(cnt) < 100:
                continue
            blob = np.zeros((H, W), np.uint8)
            cv2.drawContours(blob, [cnt], -1, 255, -1)
            sel = (blob > 0) & (mask > 0)
            pts = P[sel]
            if len(pts) < 50:
                continue
            o = self.describe(pts, table_z)
            if o is None:
                continue
            # 시야(영상 테두리)나 인식 영역(roi) 경계에 걸린 덩어리는 일부만 보인 것 → 크기를 믿을 수 없다
            x, y, w, h = cv2.boundingRect(cnt)
            at_image_edge = x <= 2 or y <= 2 or x + w >= W - 2 or y + h >= H - 2
            xs, ys = pts[:, 0], pts[:, 1]
            at_roi_edge = (xs.min() < c["roi_x"][0] + 0.005 or xs.max() > c["roi_x"][1] - 0.005 or
                           ys.min() < c["roi_y"][0] + 0.005 or ys.max() > c["roi_y"][1] - 0.005)
            if (at_image_edge or at_roi_edge) and o.graspable:   # 이미 '너무 넓음'이면 그 이유가 우선
                o.graspable = False
                o.note = "가장자리에 잘림"
            o.id = f"obj_{len(objects)}"
            objects.append((o, cnt))

        self._publish(objects, msg.header.stamp, rgb, mask, table_z)

    def describe(self, pts, table_z):
        """한 덩어리의 3D 점 → DetectedObject (모양·중심·폭·방향·높이).

        ⚠ 카메라가 물체 바로 위가 아니면 옆면도 보인다 → 전체 외곽은 길쭉하게 부푼다.
          그래서 모양·중심·방향은 **윗면 띠**(윗면에서 1.5 cm 이내 점)로 정하고,
          집을 수 있는지 판정할 폭은 전체 외곽까지 봐서 보수적으로(큰 쪽) 잡는다."""
        c = self.cfg
        h = pts[:, 2] - table_z
        top = float(np.percentile(h, 95))           # 윗면 높이 (튀는 점 5% 무시)

        hull_all = cv2.convexHull(pts[:, :2].astype(np.float32))
        if cv2.contourArea(hull_all) < c["min_area"]:   # 위에서 본 면적 [m²]이 너무 작으면 잡음
            return None
        _, (a_all, b_all), _ = cv2.minAreaRect(hull_all)

        top_pts = pts[h > top - 0.015]
        if len(top_pts) < 20:
            return None
        hull = cv2.convexHull(top_pts[:, :2].astype(np.float32))
        rect = cv2.minAreaRect(hull)
        (cx, cy), (a, b), _ = rect
        box = cv2.boxPoints(rect)
        e1, e2 = box[1] - box[0], box[2] - box[1]
        short_vec = e1 if np.linalg.norm(e1) <= np.linalg.norm(e2) else e2
        width, length = float(min(a, b)), float(max(a, b))

        fill = float(cv2.contourArea(hull)) / max(width * length, 1e-9)  # 사각형을 얼마나 채우나 (원 ≈ 0.785)
        aspect = length / max(width, 1e-9)
        o = DetectedObject()
        if aspect < 1.2 and fill < 0.88:
            o.shape = "round"
            (cx, cy), r = cv2.minEnclosingCircle(hull)
            length = float(2 * r)
            o.yaw = float("nan")                     # 원은 어느 방향으로 잡아도 같다
        else:
            o.shape = "box"
            yaw = math.atan2(float(short_vec[1]), float(short_vec[0]))
            # 그리퍼는 180° 대칭 → -90°~+90° 로 정리
            while yaw > math.pi / 2:
                yaw -= math.pi
            while yaw <= -math.pi / 2:
                yaw += math.pi
            o.yaw = yaw
        o.top_center = Point(x=float(cx), y=float(cy), z=table_z + top)
        o.width, o.length, o.height = width, max(length, width), top   # 윗면 기준 크기(표시용)
        # 집을 수 있나: 옆면까지 본 외곽 폭과 윗면 폭 중 큰 쪽으로 보수적으로 판정
        o.graspable = max(width, float(min(a_all, b_all))) <= float(c["max_grip_width"])
        if not o.graspable:
            o.note = "너무 넓음"
        return o

    # ------------------------------------------------------------------ 발행
    def _publish(self, objects, stamp, rgb, mask, table_z=None):
        out = DetectedObjectArray()
        out.header.stamp = stamp
        out.header.frame_id = self.world
        out.objects = [o for o, _ in objects]
        self.pub.publish(out)
        self.pub_mk.publish(object_markers(out.objects, self.world, table_z))

        # 디버그 그림: 컬러 영상 + 물체 마스크(초록) + 윤곽·이름
        img = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
        if mask is not None:
            img[mask > 0] = (0.5 * img[mask > 0] + [0, 110, 0]).astype(np.uint8)
        for o, cnt in objects:
            col = (0, 255, 0) if o.graspable else (0, 0, 255)
            cv2.drawContours(img, [cnt], -1, col, 2)
            x, y, _, _ = cv2.boundingRect(cnt)
            label = f"{o.id} {o.shape} {o.width * 100:.1f}cm" + ("" if o.graspable else " X")
            cv2.putText(img, label, (x, max(15, y - 5)),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, col, 2)
        if table_z is not None:
            cv2.putText(img, f"table z = {table_z:.3f} m   objects = {len(objects)}", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
        self.pub_img.publish(self.bridge.cv2_to_imgmsg(img, "bgr8"))


def main():
    rclpy.init()
    node = ObjectDetector()
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
