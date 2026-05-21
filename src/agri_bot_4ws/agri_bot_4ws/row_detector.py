#!/usr/bin/env python3
"""
row_detector.py
===============
2D LiDAR 기반 고설 베드 통로 감지 노드

발행:
  /row/state [Float32MultiArray]
    [lateral_error, heading_error, corridor_width, state, side_pillars]
    state: 0=FOLLOWING, 1=ROW_END, 2=NOT_FOUND, 3=INSIDE_BED
    side_pillars: 로봇 측면 ±0.3m 구간 기둥 수

  /row/side_targets [Float32MultiArray]
    측면 관찰 클러스터 y값 리스트

  /row/markers [MarkerArray]
    RViz2 시각화용 마커
    빨간 구체: 왼쪽 기둥
    파란 구체: 오른쪽 기둥
    녹색 선:   RANSAC 직선 (통로 방향)
    노란 선:   통로 중앙선 (lateral_error 기준)
"""

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import LaserScan
from std_msgs.msg import Float32MultiArray
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
import numpy as np
import math


class RowState:
    FOLLOWING  = 0.0
    ROW_END    = 1.0
    NOT_FOUND  = 2.0
    INSIDE_BED = 3.0


class RowDetector(Node):
    def __init__(self):
        super().__init__('row_detector')

        self.declare_parameter('max_range',        3.0)
        self.declare_parameter('cluster_eps',      0.15)
        self.declare_parameter('cluster_min_pts',  2)
        self.declare_parameter('pillar_max_pts',   15)
        self.declare_parameter('corridor_min',     0.6)
        self.declare_parameter('corridor_max',     2.0)
        self.declare_parameter('inside_bed_max',   0.5)
        self.declare_parameter('ransac_iters',     50)
        self.declare_parameter('ransac_thresh',    0.05)
        self.declare_parameter('side_obs_x_range', 0.5)
        self.declare_parameter('side_cluster_gap', 0.2)

        self.max_range        = self.get_parameter('max_range').value
        self.eps              = self.get_parameter('cluster_eps').value
        self.min_pts          = self.get_parameter('cluster_min_pts').value
        self.pillar_max       = self.get_parameter('pillar_max_pts').value
        self.corr_min         = self.get_parameter('corridor_min').value
        self.corr_max         = self.get_parameter('corridor_max').value
        self.bed_max          = self.get_parameter('inside_bed_max').value
        self.ransac_iters     = self.get_parameter('ransac_iters').value
        self.ransac_thresh    = self.get_parameter('ransac_thresh').value
        self.side_obs_x       = self.get_parameter('side_obs_x_range').value
        self.side_cluster_gap = self.get_parameter('side_cluster_gap').value

        self.last_corridor_width = 0.0

        self.pub_state  = self.create_publisher(
            Float32MultiArray, '/row/state', 10)
        self.pub_side   = self.create_publisher(
            Float32MultiArray, '/row/side_targets', 10)
        self.pub_marker = self.create_publisher(
            MarkerArray, '/row/markers', 10)

        self.create_subscription(
            LaserScan, '/scan', self.scan_callback, 10)

        self.get_logger().info('Row Detector 시작 (RViz2 마커 발행 활성화)')

    # ══════════════════════════════════════════════════════
    # 메인 스캔 콜백
    # ══════════════════════════════════════════════════════
    def scan_callback(self, msg: LaserScan):
        pts = self.scan_to_points(msg)
        if len(pts) < 4:
            self.publish_state(0.0, 0.0, self.last_corridor_width,
                               RowState.NOT_FOUND, 0)
            self.publish_markers([], [], None, 0.0)
            return

        left_pts  = pts[pts[:, 1] >  0.1]
        right_pts = pts[pts[:, 1] < -0.1]

        left_clusters  = self.cluster(left_pts)
        right_clusters = self.cluster(right_pts)
        left_pillars   = self.filter_pillars(left_clusters)
        right_pillars  = self.filter_pillars(right_clusters)

        front_left  = [c for c in left_pillars
                       if np.mean(c[:, 0]) > 0.0]
        front_right = [c for c in right_pillars
                       if np.mean(c[:, 0]) > 0.0]

        danger_left  = [c for c in left_pillars
                        if -0.3 <= np.mean(c[:, 0]) <= 0.3]
        danger_right = [c for c in right_pillars
                        if -0.3 <= np.mean(c[:, 0]) <= 0.3]
        side_pillars = len(danger_left) + len(danger_right)

        side_mask = np.abs(pts[:, 0]) < self.side_obs_x
        side_pts  = pts[side_mask]
        left_targets  = self.extract_side_clusters(
            side_pts[side_pts[:, 1] > 0.1], direction=1)
        right_targets = self.extract_side_clusters(
            side_pts[side_pts[:, 1] < -0.1], direction=-1)
        self.publish_side_targets(left_targets, right_targets)

        # 행 끝 판단
        if (len(front_left) == 0 and
                len(front_right) == 0 and
                len(left_pillars) > 0 and
                len(right_pillars) > 0):
            self.publish_state(0.0, 0.0, self.last_corridor_width,
                               RowState.ROW_END, side_pillars)
            self.publish_markers(left_pillars, right_pillars, None, 0.0)
            self.get_logger().info(
                f'state=ROW_END width={self.last_corridor_width:.2f}m '
                f'side={side_pillars}',
                throttle_duration_sec=0.3)
            return

        nearest_left  = self.nearest_cluster(left_pillars)
        nearest_right = self.nearest_cluster(right_pillars)

        state, lateral_error, heading_error, corridor_width, best_angle = \
            self.analyze(nearest_left, nearest_right,
                         left_pillars, right_pillars)

        if state == RowState.FOLLOWING:
            self.last_corridor_width = corridor_width

        self.publish_state(lateral_error, heading_error,
                           corridor_width, state, side_pillars)
        self.publish_markers(left_pillars, right_pillars,
                             best_angle, lateral_error)
        self.get_logger().info(
            f'state={self.state_name(state)} '
            f'lat={lateral_error:.3f}m '
            f'head={math.degrees(heading_error):.1f}deg '
            f'width={corridor_width:.2f}m '
            f'side={side_pillars}',
            throttle_duration_sec=0.3)

    # ══════════════════════════════════════════════════════
    # RViz2 마커 발행
    # 빨간 구체: 왼쪽 기둥 위치
    # 파란 구체: 오른쪽 기둥 위치
    # 녹색 선:   RANSAC 직선 (통로 방향)
    # 노란 선:   통로 중앙선
    # 흰색 선:   로봇 현재 위치에서 중앙까지 오차선
    # ══════════════════════════════════════════════════════
    def publish_markers(self, left_pillars, right_pillars,
                        best_angle, lateral_error):
        markers = MarkerArray()
        mid = 0

        # ── 기존 마커 전체 삭제 ────────────────────────
        delete_all = Marker()
        delete_all.action = Marker.DELETEALL
        delete_all.header.frame_id = 'base_link'
        markers.markers.append(delete_all)

        # ── 왼쪽 기둥: 빨간 구체 ──────────────────────
        for c in left_pillars:
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns   = 'left_pillars'
            m.id   = mid; mid += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(np.mean(c[:, 0]))
            m.pose.position.y = float(np.mean(c[:, 1]))
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 1.0
            m.color.r = 1.0
            m.color.g = 0.0
            m.color.b = 0.0
            m.color.a = 0.9
            markers.markers.append(m)

        # ── 오른쪽 기둥: 파란 구체 ────────────────────
        for c in right_pillars:
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns   = 'right_pillars'
            m.id   = mid; mid += 1
            m.type = Marker.CYLINDER
            m.action = Marker.ADD
            m.pose.position.x = float(np.mean(c[:, 0]))
            m.pose.position.y = float(np.mean(c[:, 1]))
            m.pose.position.z = 0.5
            m.pose.orientation.w = 1.0
            m.scale.x = 0.08
            m.scale.y = 0.08
            m.scale.z = 1.0
            m.color.r = 0.0
            m.color.g = 0.0
            m.color.b = 1.0
            m.color.a = 0.9
            markers.markers.append(m)

        # ── RANSAC 직선: 녹색 선 ──────────────────────
        # 기둥들의 중심점을 이은 직선 = 통로 방향
        if best_angle is not None:
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns   = 'ransac_line'
            m.id   = mid; mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.05
            m.color.r = 0.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 1.0
            dx = math.cos(best_angle) * 3.0
            dy = math.sin(best_angle) * 3.0
            p1 = Point(); p1.x = -dx; p1.y = -dy; p1.z = 0.1
            p2 = Point(); p2.x =  dx; p2.y =  dy; p2.z = 0.1
            m.points = [p1, p2]
            markers.markers.append(m)

        # ── 통로 중앙선: 노란 선 ──────────────────────
        # lateral_error=0이면 로봇이 중앙에 있음
        # 노란 선이 y=0을 지나면 완벽한 중앙 주행
        if best_angle is not None:
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns   = 'center_line'
            m.id   = mid; mid += 1
            m.type = Marker.LINE_STRIP
            m.action = Marker.ADD
            m.scale.x = 0.03
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 0.0
            m.color.a = 0.8
            dx = math.cos(best_angle) * 3.0
            dy = math.sin(best_angle) * 3.0
            # 중앙선은 lateral_error만큼 옆으로 이동
            offset = lateral_error
            perp_x = -math.sin(best_angle) * offset
            perp_y =  math.cos(best_angle) * offset
            p1 = Point()
            p1.x = -dx + perp_x; p1.y = -dy + perp_y; p1.z = 0.1
            p2 = Point()
            p2.x =  dx + perp_x; p2.y =  dy + perp_y; p2.z = 0.1
            m.points = [p1, p2]
            markers.markers.append(m)

        # ── lateral_error 화살표: 흰색 ────────────────
        # 로봇(원점)에서 통로 중앙까지의 오차 방향 표시
        if best_angle is not None and abs(lateral_error) > 0.01:
            m = Marker()
            m.header.frame_id = 'base_link'
            m.ns   = 'lateral_error'
            m.id   = mid; mid += 1
            m.type = Marker.ARROW
            m.action = Marker.ADD
            m.scale.x = 0.05   # 화살표 굵기
            m.scale.y = 0.10   # 화살표 머리 크기
            m.scale.z = 0.10
            m.color.r = 1.0
            m.color.g = 1.0
            m.color.b = 1.0
            m.color.a = 0.9
            perp_x = -math.sin(best_angle) * lateral_error
            perp_y =  math.cos(best_angle) * lateral_error
            p_start = Point(); p_start.x = 0.0; p_start.y = 0.0; p_start.z = 0.2
            p_end   = Point(); p_end.x = perp_x; p_end.y = perp_y; p_end.z = 0.2
            m.points = [p_start, p_end]
            markers.markers.append(m)

        # ── 왼쪽 기둥만의 RANSAC 직선: 주황색 ────────
        left_centers = [np.mean(c, axis=0) for c in left_pillars
                        if len(c) >= self.min_pts]
        if len(left_centers) >= 2:
            left_pts = np.array(left_centers)
            left_angle = self.ransac_line(left_pts)
            if left_angle is not None:
                m = Marker()
                m.header.frame_id = 'base_link'
                m.ns   = 'left_ransac'
                m.id   = mid; mid += 1
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.04
                m.color.r = 1.0
                m.color.g = 0.5
                m.color.b = 0.0
                m.color.a = 1.0
                dx = math.cos(left_angle) * 3.0
                dy = math.sin(left_angle) * 3.0
                offset_y = float(np.mean([c[1] for c in left_centers]))
                p1 = Point()
                p1.x = -dx; p1.y = -dy + offset_y; p1.z = 0.15
                p2 = Point()
                p2.x =  dx; p2.y =  dy + offset_y; p2.z = 0.15
                m.points = [p1, p2]
                markers.markers.append(m)

        # ── 오른쪽 기둥만의 RANSAC 직선: 하늘색 ──────
        right_centers = [np.mean(c, axis=0) for c in right_pillars
                         if len(c) >= self.min_pts]
        if len(right_centers) >= 2:
            right_pts = np.array(right_centers)
            right_angle = self.ransac_line(right_pts)
            if right_angle is not None:
                m = Marker()
                m.header.frame_id = 'base_link'
                m.ns   = 'right_ransac'
                m.id   = mid; mid += 1
                m.type = Marker.LINE_STRIP
                m.action = Marker.ADD
                m.scale.x = 0.04
                m.color.r = 0.0
                m.color.g = 1.0
                m.color.b = 1.0
                m.color.a = 1.0
                dx = math.cos(right_angle) * 3.0
                dy = math.sin(right_angle) * 3.0
                offset_y = float(np.mean([c[1] for c in right_centers]))
                p1 = Point()
                p1.x = -dx; p1.y = -dy + offset_y; p1.z = 0.15
                p2 = Point()
                p2.x =  dx; p2.y =  dy + offset_y; p2.z = 0.15
                m.points = [p1, p2]
                markers.markers.append(m)

        self.pub_marker.publish(markers)

    # ══════════════════════════════════════════════════════
    # analyze: best_angle도 함께 반환
    # ══════════════════════════════════════════════════════
    def analyze(self, nearest_left, nearest_right,
                left_pillars, right_pillars):
        has_left  = nearest_left  is not None
        has_right = nearest_right is not None

        if not has_left and not has_right:
            return RowState.NOT_FOUND, 0.0, 0.0, 0.0, None

        if not has_left or not has_right:
            return RowState.ROW_END, 0.0, 0.0, self.last_corridor_width, None

        left_y  = float(np.median(nearest_left[:, 1]))
        right_y = float(np.median(nearest_right[:, 1]))
        corridor_width = abs(left_y) + abs(right_y)

        if corridor_width < self.bed_max:
            return RowState.INSIDE_BED, 0.0, 0.0, corridor_width, None

        if corridor_width > self.corr_max:
            return RowState.NOT_FOUND, 0.0, 0.0, corridor_width, None

        lateral_error = (abs(right_y) - abs(left_y)) / 2.0
        heading_error, best_angle = self.calc_heading_error(
            left_pillars, right_pillars)

        return (RowState.FOLLOWING, lateral_error,
                heading_error, corridor_width, best_angle)

    # ══════════════════════════════════════════════════════
    # calc_heading_error: best_angle도 함께 반환
    # ══════════════════════════════════════════════════════
    def calc_heading_error(self, left_pillars, right_pillars):
        left_centers  = [np.mean(c, axis=0) for c in left_pillars
                         if len(c) >= self.min_pts]
        right_centers = [np.mean(c, axis=0) for c in right_pillars
                         if len(c) >= self.min_pts]
        all_centers = left_centers + right_centers
        if len(all_centers) < 2:
            return 0.0, None
        pts = np.array(all_centers)
        best_angle = self.ransac_line(pts)
        if best_angle is None:
            return 0.0, None
        heading_error = best_angle
        while heading_error >  math.pi / 2:
            heading_error -= math.pi
        while heading_error < -math.pi / 2:
            heading_error += math.pi
        return heading_error, best_angle

    # ══════════════════════════════════════════════════════
    # 나머지 함수들
    # ══════════════════════════════════════════════════════
    def extract_side_clusters(self, pts, direction):
        if len(pts) < 2:
            return []
        y_vals = np.sort(np.abs(pts[:, 1]))
        clusters = []
        current  = [y_vals[0]]
        for v in y_vals[1:]:
            if v - current[-1] < self.side_cluster_gap:
                current.append(v)
            else:
                clusters.append(float(np.mean(current)))
                current = [v]
        clusters.append(float(np.mean(current)))
        if direction > 0:
            return clusters
        else:
            return [-c for c in clusters]

    def publish_side_targets(self, left_targets, right_targets):
        data = []
        data.append(float(len(left_targets)))
        data.extend([float(v) for v in left_targets])
        data.append(float(len(right_targets)))
        data.extend([float(v) for v in right_targets])
        msg = Float32MultiArray()
        msg.data = data
        self.pub_side.publish(msg)

    def scan_to_points(self, msg: LaserScan) -> np.ndarray:
        angles = (np.arange(len(msg.ranges)) *
                  msg.angle_increment + msg.angle_min)
        ranges = np.array(msg.ranges, dtype=np.float32)
        valid  = (np.isfinite(ranges) &
                  (ranges > msg.range_min) &
                  (ranges < self.max_range))
        r = ranges[valid]
        a = angles[valid]
        return np.column_stack([r * np.cos(a), r * np.sin(a)])

    def cluster(self, points):
        if len(points) < self.min_pts:
            return []
        labels = [-1] * len(points)
        cluster_id = 0
        for i in range(len(points)):
            if labels[i] != -1:
                continue
            dists = np.linalg.norm(points - points[i], axis=1)
            neighbors = list(np.where(dists < self.eps)[0])
            if len(neighbors) < self.min_pts:
                continue
            labels[i] = cluster_id
            k = 0
            while k < len(neighbors):
                nb = neighbors[k]
                if labels[nb] == -1:
                    labels[nb] = cluster_id
                    nb_dists = np.linalg.norm(points - points[nb], axis=1)
                    nb_neighbors = list(np.where(nb_dists < self.eps)[0])
                    if len(nb_neighbors) >= self.min_pts:
                        for n in nb_neighbors:
                            if n not in neighbors:
                                neighbors.append(n)
                k += 1
            cluster_id += 1
        clusters = []
        for cid in range(cluster_id):
            idxs = [i for i, l in enumerate(labels) if l == cid]
            clusters.append(points[idxs])
        return clusters

    def filter_pillars(self, clusters):
        return [c for c in clusters
                if self.min_pts <= len(c) <= self.pillar_max]

    def nearest_cluster(self, clusters):
        if not clusters:
            return None
        return min(clusters,
                   key=lambda c: np.mean(np.linalg.norm(c, axis=1)))

    def ransac_line(self, pts):
        if len(pts) < 2:
            return None
        best_inliers = 0
        best_angle   = None
        for _ in range(self.ransac_iters):
            idx = np.random.choice(len(pts), 2, replace=False)
            p1, p2 = pts[idx[0]], pts[idx[1]]
            dx = p2[0] - p1[0]
            dy = p2[1] - p1[1]
            if math.sqrt(dx**2 + dy**2) < 0.01:
                continue
            a = dy; b = -dx
            c = -(a * p1[0] + b * p1[1])
            norm = math.sqrt(a**2 + b**2)
            dists = np.abs(a * pts[:, 0] + b * pts[:, 1] + c) / norm
            inliers = np.sum(dists < self.ransac_thresh)
            if inliers > best_inliers:
                best_inliers = inliers
                best_angle = math.atan2(dy, dx)
        return best_angle

    def publish_state(self, lateral_error, heading_error,
                      corridor_width, state, side_pillars=0):
        msg = Float32MultiArray()
        msg.data = [float(lateral_error), float(heading_error),
                    float(corridor_width), float(state),
                    float(side_pillars)]
        self.pub_state.publish(msg)

    def state_name(self, state):
        return {RowState.FOLLOWING: 'FOLLOWING',
                RowState.ROW_END:   'ROW_END',
                RowState.NOT_FOUND: 'NOT_FOUND',
                RowState.INSIDE_BED:'INSIDE_BED'}.get(state, '?')


def main(args=None):
    rclpy.init(args=args)
    node = RowDetector()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        pass
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()