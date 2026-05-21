#!/usr/bin/env python3
"""
corridor_detector.py
====================
Depth 카메라로 통로를 감지하는 노드

동작 원리:
  고설 베드 구조:
    - 베드 상판: 카메라에서 가까움 (작은 depth값)
    - 통로 바닥: 카메라에서 멀음 (큰 depth값)
    - 기둥: 중간 depth값

  Depth 이미지에서:
    - 먼 영역(바닥) = 통로
    - 가까운 영역(상판) = 베드
    - 통로 경계선 → 소실점 계산 → 방향 오차

토픽:
  sub: /camera_front/depth/image_raw
  sub: /camera_rear/depth/image_raw
  pub: /corridor/front/state  (std_msgs/Float32MultiArray)
  pub: /corridor/rear/state
  pub: /corridor/front/debug  (sensor_msgs/Image)
  pub: /corridor/rear/debug

발행 데이터 [lateral_error, heading_error, corridor_ratio, state]:
  lateral_error:  횡방향 오차 (픽셀, 양수=오른쪽 치우침)
  heading_error:  방향 오차 (픽셀, 양수=오른쪽 방향)
  corridor_ratio: 통로 픽셀 비율 (0.0~1.0)
  state:          0=CLEAR, 1=END, 2=NOT_FOUND
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import Image
from std_msgs.msg import Float32MultiArray
import numpy as np
import cv2
from cv_bridge import CvBridge


class CorridorState:
    CLEAR     = 0.0   # 통로 정상
    END       = 1.0   # 행 끝
    NOT_FOUND = 2.0   # 통로 없음 (오배치)


class CorridorDetector(Node):
    def __init__(self):
        super().__init__('corridor_detector')

        # ── 파라미터 ─────────────────────────────────────────
        self.declare_parameter('floor_depth_min', 0.6)   # 바닥 최소 거리 (m)
        self.declare_parameter('floor_depth_max', 2.0)   # 바닥 최대 거리 (m)
        self.declare_parameter('bed_depth_max',   0.5)   # 베드 최대 거리 (m)
        self.declare_parameter('end_threshold',   0.15)  # 행 끝 판단 비율
        self.declare_parameter('found_threshold', 0.25)  # 통로 발견 판단 비율
        self.declare_parameter('debug',           True)  # 디버그 이미지 발행

        self.floor_min  = self.get_parameter('floor_depth_min').value
        self.floor_max  = self.get_parameter('floor_depth_max').value
        self.bed_max    = self.get_parameter('bed_depth_max').value
        self.end_thr    = self.get_parameter('end_threshold').value
        self.found_thr  = self.get_parameter('found_threshold').value
        self.debug      = self.get_parameter('debug').value

        self.bridge = CvBridge()

        # ── Publishers ────────────────────────────────────────
        self.front_pub = self.create_publisher(
            Float32MultiArray, '/corridor/front/state', 10)
        self.rear_pub  = self.create_publisher(
            Float32MultiArray, '/corridor/rear/state', 10)

        if self.debug:
            self.front_debug_pub = self.create_publisher(
                Image, '/corridor/front/debug', 10)
            self.rear_debug_pub  = self.create_publisher(
                Image, '/corridor/rear/debug', 10)

        # ── Subscribers ───────────────────────────────────────
        self.create_subscription(
            Image, '/camera_front/depth/image_raw',
            lambda msg: self.depth_callback(msg, 'front'), 10)
        self.create_subscription(
            Image, '/camera_rear/depth/image_raw',
            lambda msg: self.depth_callback(msg, 'rear'), 10)

        self.get_logger().info('Corridor Detector 시작')
        self.get_logger().info(
            f'바닥 범위: {self.floor_min}~{self.floor_max}m | '
            f'베드: <{self.bed_max}m')

    def depth_callback(self, msg: Image, camera: str):
        try:
            # depth 이미지 변환 (32FC1 또는 16UC1)
            if msg.encoding == '32FC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '32FC1')
            elif msg.encoding == '16UC1':
                depth = self.bridge.imgmsg_to_cv2(msg, '16UC1').astype(np.float32)
                depth /= 1000.0  # mm → m
            else:
                depth = self.bridge.imgmsg_to_cv2(msg)
                depth = depth.astype(np.float32)

            result = self.detect_corridor(depth, camera)

            # 발행
            state_msg = Float32MultiArray()
            state_msg.data = [
                float(result['lateral_error']),
                float(result['heading_error']),
                float(result['corridor_ratio']),
                float(result['state'])
            ]

            if camera == 'front':
                self.front_pub.publish(state_msg)
                if self.debug and hasattr(self, 'front_debug_pub'):
                    debug_img = self.bridge.cv2_to_imgmsg(
                        result['debug_img'], 'bgr8')
                    self.front_debug_pub.publish(debug_img)
            else:
                self.rear_pub.publish(state_msg)
                if self.debug and hasattr(self, 'rear_debug_pub'):
                    debug_img = self.bridge.cv2_to_imgmsg(
                        result['debug_img'], 'bgr8')
                    self.rear_debug_pub.publish(debug_img)

        except Exception as e:
            self.get_logger().warn(f'depth_callback 오류: {e}')

    def detect_corridor(self, depth: np.ndarray, camera: str) -> dict:
        """
        Depth 이미지에서 통로 감지

        Returns:
            dict with keys:
                lateral_error: 횡방향 오차 (픽셀)
                heading_error: 방향 오차 (픽셀)
                corridor_ratio: 통로 픽셀 비율
                state: CorridorState
                debug_img: 디버그 이미지
        """
        h, w = depth.shape

        # ── 1. 하단 60% ROI 사용 (바닥 영역) ─────────────────
        roi_top = int(h * 0.4)
        depth_roi = depth[roi_top:h, :]

        # ── 2. 유효 depth 필터 ────────────────────────────────
        valid_mask = (depth_roi > 0.01) & np.isfinite(depth_roi)

        # ── 3. 바닥(통로) 마스크 ─────────────────────────────
        # 고설 베드: 통로 바닥은 카메라에서 멀리 있음
        floor_mask = (
            valid_mask &
            (depth_roi >= self.floor_min) &
            (depth_roi <= self.floor_max)
        )

        # ── 4. 통로 픽셀 비율 계산 ───────────────────────────
        total_pixels    = depth_roi.size
        corridor_pixels = np.sum(floor_mask)
        corridor_ratio  = corridor_pixels / total_pixels

        # ── 5. 상태 판단 ─────────────────────────────────────
        if corridor_ratio < self.end_thr:
            if corridor_ratio < 0.05:
                state = CorridorState.NOT_FOUND
            else:
                state = CorridorState.END
        else:
            state = CorridorState.CLEAR

        # ── 6. 횡방향 오차 계산 ──────────────────────────────
        lateral_error = 0.0
        heading_error = 0.0

        if state == CorridorState.CLEAR:
            # 통로 픽셀의 무게중심 계산
            cols = np.where(floor_mask)[1]
            if len(cols) > 0:
                corridor_center = float(np.mean(cols))
                image_center    = w / 2.0
                lateral_error   = corridor_center - image_center
                # 양수 = 통로 중심이 이미지 오른쪽 = 로봇이 왼쪽으로 치우침

            # 소실점 계산 (Hough 라인)
            heading_error = self.calc_vanishing_point(floor_mask, w)

        # ── 7. 디버그 이미지 ──────────────────────────────────
        debug_img = self.make_debug_image(
            depth_roi, floor_mask, lateral_error,
            corridor_ratio, state, w, h - roi_top, camera)

        self.get_logger().info(
            f'[{camera}] ratio={corridor_ratio:.2f} '
            f'lat={lateral_error:.1f}px '
            f'head={heading_error:.1f}px '
            f'state={state}',
            throttle_duration_sec=0.5
        )

        return {
            'lateral_error':  lateral_error,
            'heading_error':  heading_error,
            'corridor_ratio': corridor_ratio,
            'state':          state,
            'debug_img':      debug_img
        }

    def calc_vanishing_point(self, floor_mask: np.ndarray, w: int) -> float:
        """
        통로 경계선에서 소실점 x좌표 계산
        소실점 x - 이미지 중앙 = heading_error
        """
        try:
            # 마스크를 uint8로 변환
            mask_u8 = floor_mask.astype(np.uint8) * 255

            # 엣지 검출
            edges = cv2.Canny(mask_u8, 50, 150)

            # Hough 라인 검출
            lines = cv2.HoughLinesP(
                edges, 1, np.pi / 180,
                threshold=30,
                minLineLength=30,
                maxLineGap=20
            )

            if lines is None:
                return 0.0

            # 수직에 가까운 선만 선택 (통로 경계선)
            h = floor_mask.shape[0]
            vp_xs = []

            for line in lines:
                x1, y1, x2, y2 = line[0]
                if abs(x2 - x1) < 1:
                    continue
                angle = abs(np.degrees(np.arctan2(y2 - y1, x2 - x1)))
                # 수직에 가까운 선 (60~120도)
                if 60 < angle < 120:
                    # 소실점 x 추정 (선을 이미지 상단으로 연장)
                    if y2 != y1:
                        slope = (x2 - x1) / (y2 - y1)
                        vp_x  = x1 + slope * (0 - y1)
                        vp_xs.append(vp_x)

            if len(vp_xs) == 0:
                return 0.0

            vanishing_x   = float(np.median(vp_xs))
            heading_error = vanishing_x - w / 2.0
            return heading_error

        except Exception:
            return 0.0

    def make_debug_image(self, depth_roi, floor_mask,
                         lateral_error, corridor_ratio,
                         state, w, h, camera):
        """디버그용 컬러 이미지 생성"""
        # depth를 0~255로 정규화
        depth_norm = cv2.normalize(
            depth_roi, None, 0, 255,
            cv2.NORM_MINMAX, dtype=cv2.CV_8U)
        debug = cv2.cvtColor(depth_norm, cv2.COLOR_GRAY2BGR)

        # 통로 영역 초록색으로 표시
        debug[floor_mask] = (0, 200, 0)

        # 중앙선
        cv2.line(debug, (w // 2, 0), (w // 2, h), (255, 255, 0), 1)

        # 통로 중심선
        if abs(lateral_error) > 0:
            cx = int(w // 2 + lateral_error)
            cx = max(0, min(w - 1, cx))
            cv2.line(debug, (cx, 0), (cx, h), (0, 0, 255), 2)

        # 상태 텍스트
        state_names = {
            CorridorState.CLEAR:     'CLEAR',
            CorridorState.END:       'ROW_END',
            CorridorState.NOT_FOUND: 'NOT_FOUND'
        }
        state_colors = {
            CorridorState.CLEAR:     (0, 255, 0),
            CorridorState.END:       (0, 165, 255),
            CorridorState.NOT_FOUND: (0, 0, 255)
        }
        cv2.putText(
            debug,
            f'{camera} {state_names.get(state, "?")} '
            f'r={corridor_ratio:.2f} lat={lateral_error:.0f}',
            (5, 20), cv2.FONT_HERSHEY_SIMPLEX,
            0.5, state_colors.get(state, (255, 255, 255)), 1
        )

        return debug


def main(args=None):
    rclpy.init(args=args)
    node = CorridorDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()
