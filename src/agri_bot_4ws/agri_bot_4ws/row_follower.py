#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import Float32MultiArray
from sensor_msgs.msg import LaserScan
from geometry_msgs.msg import Twist
from std_msgs.msg import String, Bool
from nav_msgs.msg import Odometry
import math
import numpy as np
from enum import Enum


def euler_from_quaternion(q):
    x, y, z, w = q
    siny_cosp = 2.0 * (w * z + x * y)
    cosy_cosp = 1.0 - 2.0 * (y * y + z * z)
    yaw = math.atan2(siny_cosp, cosy_cosp)
    return 0.0, 0.0, yaw


class State(Enum):
    FOLLOWING             = "FOLLOWING"
    ROW_END               = "ROW_END"
    PIVOT_FWD             = "PIVOT_FWD"
    PIVOT_CRAB            = "PIVOT_CRAB"
    PIVOT_TURN            = "PIVOT_TURN"
    PIVOT_ALIGN           = "PIVOT_ALIGN"
    LAST_ROW              = "LAST_ROW"
    RETURN_TURN           = "RETURN_TURN"
    RETURN_CRAB           = "RETURN_CRAB"
    RETURN_CORRIDOR_ALIGN = "RETURN_CORRIDOR_ALIGN"  # 크랩 기반 통로 중앙 보정
    RETURN_ALIGN          = "RETURN_ALIGN"
    RETURN_FWD            = "RETURN_FWD"
    MISSION_DONE          = "MISSION_DONE"
    STOPPED               = "STOPPED"


class RowFollower(Node):
    def __init__(self):
        super().__init__('row_follower')

        self.declare_parameter('linear_vel',            0.3)
        self.declare_parameter('max_angular_vel',       0.5)
        self.declare_parameter('target_dist',          -1.0)
        self.declare_parameter('Kp',                    1.2)
        self.declare_parameter('Ki',                    0.01)
        self.declare_parameter('Kd',                    0.3)
        self.declare_parameter('row_end_dist',          1.2)
        self.declare_parameter('obstacle_dist',         0.4)
        self.declare_parameter('side_angle_range',      30)
        self.declare_parameter('last_row_confirm',      5)
        self.declare_parameter('corridor_interval_tol', 0.3)

        self.linear_vel       = self.get_parameter('linear_vel').value
        self.max_angular_vel  = self.get_parameter('max_angular_vel').value
        self.target_dist      = self.get_parameter('target_dist').value
        self.Kp               = self.get_parameter('Kp').value
        self.Ki               = self.get_parameter('Ki').value
        self.Kd               = self.get_parameter('Kd').value
        self.row_end_dist     = self.get_parameter('row_end_dist').value
        self.obstacle_dist    = self.get_parameter('obstacle_dist').value
        self.side_angle_range = self.get_parameter('side_angle_range').value
        self.LAST_ROW_CONFIRM = self.get_parameter('last_row_confirm').value
        self.corridor_tol     = self.get_parameter('corridor_interval_tol').value

        # FSM
        self.state            = State.FOLLOWING
        self.current_row      = 1
        self.pid_integral     = 0.0
        self.pid_prev_error   = 0.0
        self.robot_yaw        = 0.0
        self.odom_x           = 0.0
        self.odom_y           = 0.0
        self.last_scan        = None
        self.row_end_count    = 0
        self.ROW_END_CONFIRM  = 5

        # spawn 위치
        self.spawn_odom_x     = None
        self.spawn_odom_y     = None

        # 행 수 자동 감지
        self.detected_max_row    = 0
        self.last_row_confidence = 0
        self.known_interval      = None

        # 피벗
        self.pivot_target_yaw    = 0.0
        self.pivot_direction     = 1

        # PIVOT_FWD
        self.fwd_no_pillar_count = 0
        self.FWD_REAR_CONFIRM    = 3

        # 크랩이동 안전확보
        self.safe_fwd_start_x    = None
        self.safe_fwd_distance   = 0.25

        # 크랩
        self.crab_target_y       = 0.0
        self.crab_target_set     = False
        self.crab_speed          = 0.2
        self.crab_arrive_thresh  = 0.05

        # ── Waypoint Return ────────────────────────────
        self.visited_corridor_ys  = []
        self.return_waypoint_idx  = 0

        # 귀환
        self.return_row           = 0
        self.return_align_count   = 0
        self.RETURN_ALIGN_CONFIRM = 3

        # RETURN_CORRIDOR_ALIGN (크랩 기반)
        self.return_corridor_align_count   = 0
        self.RETURN_CORRIDOR_ALIGN_CONFIRM = 5
        self.RETURN_CORRIDOR_ALIGN_THRESH  = 0.05

        # PIVOT_ALIGN
        self.align_count      = 0
        self.ALIGN_CONFIRM    = 5
        self.ALIGN_THRESHOLD  = math.radians(2.0)
        self.crab_min_width   = 1.2

        self.side_targets_raw = []

        self.cmd_pub     = self.create_publisher(Twist,  '/cmd_vel', 10)
        self.status_pub  = self.create_publisher(String, '/row_follower/status', 10)
        self.sprayer_pub = self.create_publisher(Bool,   '/sprayer/cmd', 10)

        self.row_state = [0.0, 0.0, 0.0, 0.0, 0.0]
        self.create_subscription(Float32MultiArray, '/row/state',
                                 self.row_state_callback, 10)
        self.create_subscription(Float32MultiArray, '/row/side_targets',
                                 self.side_targets_callback, 10)
        self.create_subscription(LaserScan, '/scan', self.scan_callback, 10)
        self.create_subscription(Odometry,  '/odom', self.odom_callback, 10)

        self.create_timer(0.05, self.control_loop)
        self.get_logger().info('Row Follower 시작 | Waypoint Return v3')

    def row_state_callback(self, msg):
        if len(msg.data) >= 5:
            self.row_state = list(msg.data)
        elif len(msg.data) >= 4:
            self.row_state = list(msg.data) + [0.0]

    def side_targets_callback(self, msg):
        self.side_targets_raw = list(msg.data)

    def scan_callback(self, msg):
        self.last_scan = msg

    def odom_callback(self, msg):
        q = msg.pose.pose.orientation
        _, _, self.robot_yaw = euler_from_quaternion([q.x, q.y, q.z, q.w])
        self.odom_x = msg.pose.pose.position.x
        self.odom_y = msg.pose.pose.position.y
        if self.spawn_odom_x is None:
            self.spawn_odom_x = self.odom_x
            self.spawn_odom_y = self.odom_y
            self.get_logger().info(
                f'📍 spawn 저장: x={self.spawn_odom_x:.2f}m y={self.spawn_odom_y:.2f}m')

    def parse_side_targets(self):
        data = self.side_targets_raw
        if len(data) < 2:
            return [], []
        idx = 0
        left_count = int(data[idx]); idx += 1
        left_targets = []
        for _ in range(left_count):
            if idx < len(data):
                left_targets.append(data[idx]); idx += 1
        if idx >= len(data):
            return left_targets, []
        right_count = int(data[idx]); idx += 1
        right_targets = []
        for _ in range(right_count):
            if idx < len(data):
                right_targets.append(data[idx]); idx += 1
        return left_targets, right_targets

    def _select_targets(self):
        left_targets, right_targets = self.parse_side_targets()
        if abs(self.robot_yaw) < math.pi / 2:
            targets = left_targets if self.pivot_direction > 0 else right_targets
        else:
            targets = right_targets if self.pivot_direction < 0 else left_targets
        return sorted(targets, key=abs)

    def get_sector_distance(self, scan, angle_center_deg, angle_range_deg):
        angle_min  = scan.angle_min
        angle_inc  = scan.angle_increment
        ranges     = np.array(scan.ranges)
        n          = len(ranges)
        center_rad = math.radians(angle_center_deg)
        range_rad  = math.radians(angle_range_deg)
        idx_center = int((center_rad - angle_min) / angle_inc) % n
        idx_range  = int(range_rad / angle_inc)
        indices    = [(idx_center + i) % n for i in range(-idx_range, idx_range + 1)]
        sector = ranges[indices]
        valid  = sector[np.isfinite(sector) &
                        (sector > scan.range_min) &
                        (sector < scan.range_max)]
        return float(np.median(valid)) if len(valid) > 0 else float('inf')

    def get_front_distance(self, scan):
        return self.get_sector_distance(scan, 0.0, 15.0)

    def stop_robot(self):
        self.cmd_pub.publish(Twist())

    def sprayer_control(self, on):
        msg = Bool(); msg.data = on
        self.sprayer_pub.publish(msg)
        self.get_logger().info(f'💧 노즐 {"ON" if on else "OFF"}')

    def publish_status(self):
        msg = String()
        msg.data = (f'state={self.state.value} row={self.current_row} '
                    f'max={self.detected_max_row} yaw={math.degrees(self.robot_yaw):.1f}°')
        self.status_pub.publish(msg)

    # ══════════════════════════════════════════════════════
    # 메인 FSM
    # ══════════════════════════════════════════════════════
    def control_loop(self):
        if self.last_scan is None:
            return
        scan = self.last_scan
        if self.state == State.MISSION_DONE:
            self.stop_robot()
            return

        front_dist = self.get_front_distance(scan)

        if self.state == State.FOLLOWING:
            if front_dist < self.obstacle_dist:
                self.get_logger().warn(f'⚠️ 장애물! {front_dist:.2f}m')
                self.state = State.STOPPED
                self.stop_robot()
                return
            self._do_following(front_dist)
        elif self.state == State.STOPPED:
            if front_dist > self.obstacle_dist + 0.2:
                self.state = State.FOLLOWING
            else:
                self.stop_robot()
        elif self.state == State.ROW_END:              self._do_row_end()
        elif self.state == State.PIVOT_FWD:            self._do_pivot_fwd()
        elif self.state == State.PIVOT_CRAB:           self._do_pivot_crab()
        elif self.state == State.PIVOT_TURN:           self._do_pivot_turn()
        elif self.state == State.PIVOT_ALIGN:          self._do_pivot_align()
        elif self.state == State.LAST_ROW:             self._do_last_row()
        elif self.state == State.RETURN_TURN:          self._do_return_turn()
        elif self.state == State.RETURN_CRAB:          self._do_return_crab()
        elif self.state == State.RETURN_CORRIDOR_ALIGN:self._do_return_corridor_align()
        elif self.state == State.RETURN_ALIGN:         self._do_return_align()
        elif self.state == State.RETURN_FWD:           self._do_return_fwd()

        self.publish_status()

    # ══════════════════════════════════════════════════════
    # FOLLOWING
    # ══════════════════════════════════════════════════════
    def _do_following(self, front_dist):
        lateral_error  = self.row_state[0]
        heading_error  = self.row_state[1]
        corridor_width = self.row_state[2]
        row_state_val  = self.row_state[3]

        if row_state_val == 1.0:
            self.row_end_count += 1
            if self.row_end_count >= self.ROW_END_CONFIRM:
                self.get_logger().info(f'🔚 {self.current_row}행 끝!')
                self.state = State.ROW_END
                self.row_end_count = 0
                self.sprayer_control(False)
                self.stop_robot()
                return
        else:
            self.row_end_count = 0

        DEAD_ZONE = 0.15
        if abs(lateral_error) > DEAD_ZONE:
            error = lateral_error - math.copysign(DEAD_ZONE, lateral_error)
            vy = max(-0.15, min(0.15, self.Kp * error))
        else:
            vy = 0.0
        wz = max(-self.max_angular_vel, min(self.max_angular_vel, heading_error * 0.5))

        cmd = Twist()
        cmd.linear.x  = self.linear_vel
        cmd.linear.y  = vy
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)
        self.get_logger().info(
            f'[FOLLOWING] {self.current_row}행 lat={lateral_error:.3f}m '
            f'vy={vy:.3f} wz={wz:.3f} width={corridor_width:.2f}m',
            throttle_duration_sec=0.5)

    # ══════════════════════════════════════════════════════
    # ROW_END
    # ══════════════════════════════════════════════════════
    def _do_row_end(self):
        self.pivot_direction = (1 if abs(self.robot_yaw) < math.pi / 2 else -1)
        self.fwd_no_pillar_count = 0
        self.crab_target_set     = False
        self.last_row_confidence = 0
        self.state = State.PIVOT_FWD
        self.safe_fwd_start_x = None
        self.get_logger().info(
            f'↩️  {self.current_row}행 → PIVOT_FWD '
            f'(크랩방향={self.pivot_direction} yaw={math.degrees(self.robot_yaw):.1f}°)')

    # ══════════════════════════════════════════════════════
    # PIVOT_FWD
    # ══════════════════════════════════════════════════════
    def _do_pivot_fwd(self):
        side_pillars   = int(self.row_state[4])
        row_state_val  = self.row_state[3]
        corridor_width = self.row_state[2]

        if not self.crab_target_set:
            targets_sorted = self._select_targets()

            if len(targets_sorted) < 2:
                if corridor_width > self.crab_min_width:
                    self.last_row_confidence += 1
                    self.get_logger().info(
                        f'🔍 마지막 행 후보 '
                        f'({self.last_row_confidence}/{self.LAST_ROW_CONFIRM}) '
                        f'cluster={len(targets_sorted)}',
                        throttle_duration_sec=0.3)
                    if self.last_row_confidence >= self.LAST_ROW_CONFIRM:
                        self.detected_max_row = self.current_row
                        self.stop_robot()
                        self.state = State.LAST_ROW
                        self.get_logger().info(f'🏁 마지막 행 확정! 총 {self.detected_max_row}행')
                        return
                else:
                    self.last_row_confidence = 0
            else:
                self.last_row_confidence = 0
                current_pillar_y  = targets_sorted[0]
                next_pillar_y     = targets_sorted[1]
                corridor_interval = next_pillar_y - current_pillar_y

                if self.known_interval is None:
                    self.known_interval = abs(corridor_interval)
                    self.get_logger().info(f'📏 통로 간격 학습: {self.known_interval:.2f}m')

                valid = (self.known_interval is None or
                         abs(abs(corridor_interval) - self.known_interval) <= self.corridor_tol)

                if valid:
                    self.crab_target_y = (
                        self.odom_y + corridor_interval * self.pivot_direction)
                    self.crab_target_set = True
                    self.get_logger().info(
                        f'🎯 크랩 목표 | 현재={current_pillar_y:.2f}m '
                        f'다음={next_pillar_y:.2f}m 간격={corridor_interval:.2f}m '
                        f'odom목표={self.crab_target_y:.2f}m')
                else:
                    self.get_logger().warn(
                        f'⚠️ 비정상 간격 {corridor_interval:.2f}m '
                        f'(예상 ±{self.corridor_tol:.1f}m)',
                        throttle_duration_sec=0.5)

        if (self.crab_target_set and row_state_val == 1.0 and side_pillars == 0):
            self.fwd_no_pillar_count += 1
            self.get_logger().info(
                f'✅ 측면 안전 ({self.fwd_no_pillar_count}/{self.FWD_REAR_CONFIRM})',
                throttle_duration_sec=0.3)

            if self.fwd_no_pillar_count >= self.FWD_REAR_CONFIRM:
                if self.safe_fwd_start_x is None:
                    self.safe_fwd_start_x = self.odom_x
                    self.get_logger().info(
                        f'📍 안전 확인 위치: x={self.safe_fwd_start_x:.2f}m '
                        f'추가 전진 목표: {self.safe_fwd_distance:.2f}m')

                extra_fwd = abs(self.odom_x - self.safe_fwd_start_x)
                if extra_fwd >= self.safe_fwd_distance:
                    self.stop_robot()
                    self.state = State.PIVOT_CRAB
                    self.get_logger().info(
                        f'✅ 크랩 시작! 추가전진={extra_fwd:.2f}m '
                        f'{self.odom_y:.2f}m → {self.crab_target_y:.2f}m')
                else:
                    cmd = Twist()
                    cmd.linear.x = self.linear_vel
                    self.cmd_pub.publish(cmd)
                    self.get_logger().info(
                        f'➡️  추가 전진 중 {extra_fwd:.2f}m / {self.safe_fwd_distance:.2f}m',
                        throttle_duration_sec=0.3)
        else:
            self.fwd_no_pillar_count = 0
            self.safe_fwd_start_x = None
            cmd = Twist()
            cmd.linear.x = self.linear_vel
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f'➡️  전진 | side={side_pillars} target_set={self.crab_target_set}',
                throttle_duration_sec=0.5)

    # ══════════════════════════════════════════════════════
    # PIVOT_CRAB
    # ══════════════════════════════════════════════════════
    def _do_pivot_crab(self):
        distance_to_goal = self.crab_target_y - self.odom_y
        remaining = abs(distance_to_goal)
        self.get_logger().info(
            f'[CRAB] y={self.odom_y:.3f}m →{self.crab_target_y:.3f}m 남은={remaining:.3f}m',
            throttle_duration_sec=0.3)

        if remaining <= self.crab_arrive_thresh:
            self.stop_robot()
            self.get_logger().info(f'✅ 크랩 완료! (오차={remaining*100:.1f}cm)')
            self.pivot_target_yaw = math.atan2(
                math.sin(self.robot_yaw + math.pi),
                math.cos(self.robot_yaw + math.pi))
            self.state = State.PIVOT_TURN
            return

        world_direction = 1 if distance_to_goal > 0 else -1
        robot_vy = (self.crab_speed * (-world_direction)
                    if abs(self.robot_yaw) > math.pi / 2
                    else self.crab_speed * world_direction)
        cmd = Twist()
        cmd.linear.y = robot_vy
        self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # PIVOT_TURN
    # ══════════════════════════════════════════════════════
    def _do_pivot_turn(self):
        yaw_error = math.atan2(
            math.sin(self.pivot_target_yaw - self.robot_yaw),
            math.cos(self.pivot_target_yaw - self.robot_yaw))
        self.get_logger().info(
            f'피벗 yaw={math.degrees(self.robot_yaw):.1f}° err={math.degrees(yaw_error):.1f}°',
            throttle_duration_sec=0.3)

        if abs(yaw_error) < math.radians(1.0):
            self.stop_robot()
            self.align_count = 0
            self.state = State.PIVOT_ALIGN
            self.get_logger().info('✅ 피벗 완료 → PIVOT_ALIGN')
            return

        cmd = Twist()
        cmd.angular.z = math.copysign(0.4, yaw_error)
        self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # PIVOT_ALIGN + Waypoint 저장
    # ══════════════════════════════════════════════════════
    def _do_pivot_align(self):
        heading_error  = self.row_state[1]
        corridor_width = self.row_state[2]

        if corridor_width < self.crab_min_width:
            self.stop_robot()
            self.get_logger().info('⏳ 통로 감지 대기...', throttle_duration_sec=0.5)
            return

        h = heading_error
        if abs(h) > math.pi / 2:
            h -= math.copysign(math.pi, h)

        if abs(h) < self.ALIGN_THRESHOLD:
            self.align_count += 1
            self.get_logger().info(
                f'🎯 정렬 ({self.align_count}/{self.ALIGN_CONFIRM}) head={math.degrees(h):.1f}°',
                throttle_duration_sec=0.3)
            if self.align_count >= self.ALIGN_CONFIRM:
                self.stop_robot()
                self.current_row += 1
                self.target_dist  = -1.0
                self.pid_integral = 0.0

                # Waypoint 저장
                self.visited_corridor_ys.append(self.odom_y)
                self.get_logger().info(
                    f'📌 Waypoint 저장 | 통로{len(self.visited_corridor_ys)} '
                    f'y={self.odom_y:.3f}m | '
                    f'전체={[f"{v:.2f}" for v in self.visited_corridor_ys]}')

                self.state = State.FOLLOWING
                self.sprayer_control(True)
                self.get_logger().info(f'✅ 정렬 완료 | {self.current_row}행 진입')
        else:
            self.align_count = 0
            wz = max(-0.2, min(0.2, h * 0.5))
            cmd = Twist()
            cmd.angular.z = wz
            self.cmd_pub.publish(cmd)
            self.get_logger().info(
                f'🔧 방향 보정 head={math.degrees(h):.1f}° wz={wz:.3f}',
                throttle_duration_sec=0.3)

    # ══════════════════════════════════════════════════════
    # LAST_ROW: 수정 - return_waypoint_idx를 len-2부터 시작
    # ══════════════════════════════════════════════════════
    def _do_last_row(self):
        self.return_row = self.current_row - 1

        # visited = [통로1_y, 통로2_y, 통로3_y]  (4행 기준)
        # 마지막 waypoint(통로3_y)는 현재 행 진입 좌표
        # 귀환 첫 크랩 목표는 통로2_y (len-2)
        self.return_waypoint_idx = len(self.visited_corridor_ys) - 2

        if len(self.visited_corridor_ys) >= 2:
            self.crab_target_y = self.visited_corridor_ys[self.return_waypoint_idx]
        elif len(self.visited_corridor_ys) == 1:
            # 2행만 있는 경우 → spawn으로 바로
            self.crab_target_y = self.spawn_odom_y
        else:
            # fallback
            spawn_direction = 1 if self.spawn_odom_y > self.odom_y else -1
            if self.known_interval is not None:
                self.crab_target_y = self.odom_y + self.known_interval * spawn_direction
            else:
                self.crab_target_y = self.spawn_odom_y

        self.state = State.RETURN_CRAB
        self.get_logger().info(
            f'🔄 귀환 시작 | {self.detected_max_row}행 완주 '
            f'| 필요 크랩={self.return_row}회 '
            f'| 현재y={self.odom_y:.2f}m '
            f'| 첫목표y={self.crab_target_y:.2f}m '
            f'| Waypoints={[f"{v:.2f}" for v in self.visited_corridor_ys]}')

    # ══════════════════════════════════════════════════════
    # RETURN_CRAB: Waypoint 기반 역방향 크랩
    # ══════════════════════════════════════════════════════
    def _do_return_crab(self):
        distance_to_goal = self.crab_target_y - self.odom_y
        remaining = abs(distance_to_goal)

        self.get_logger().info(
            f'[RETURN_CRAB] y={self.odom_y:.3f}m →{self.crab_target_y:.3f}m '
            f'남은={remaining:.3f}m 남은크랩={self.return_row}',
            throttle_duration_sec=0.3)

        if remaining <= self.crab_arrive_thresh:
            self.return_row -= 1
            self.return_waypoint_idx -= 1
            self.stop_robot()
            self.get_logger().info(
                f'✅ 귀환 크랩 완료 | 남은 크랩={self.return_row}회 '
                f'| 현재y={self.odom_y:.2f}m')

            if self.return_row <= 0:
                y_error = abs(self.odom_y - self.spawn_odom_y)
                self.get_logger().info(f'🏠 통로1 도달 | spawn_y 오차={y_error*100:.1f}cm')
                self.return_align_count = 0
                self.state = State.RETURN_ALIGN
            else:
                # 다음 Waypoint 설정
                if (self.return_waypoint_idx >= 0 and
                        self.return_waypoint_idx < len(self.visited_corridor_ys)):
                    self.crab_target_y = self.visited_corridor_ys[self.return_waypoint_idx]
                    self.get_logger().info(
                        f'📍 다음 Waypoint: idx={self.return_waypoint_idx} '
                        f'y={self.crab_target_y:.2f}m')
                else:
                    # 마지막: spawn_y로
                    self.crab_target_y = self.spawn_odom_y
                    self.get_logger().info(f'📍 마지막 크랩 → spawn_y={self.spawn_odom_y:.2f}m')

                # 크랩 전 통로 중앙 보정 (크랩 기반)
                self.return_corridor_align_count = 0
                self.state = State.RETURN_CORRIDOR_ALIGN
                self.get_logger().info(
                    f'🔧 통로 중앙 보정 시작 | 다음목표={self.crab_target_y:.2f}m')
            return

        world_direction = 1 if distance_to_goal > 0 else -1
        robot_vy = (self.crab_speed * (-world_direction)
                    if abs(self.robot_yaw) > math.pi / 2
                    else self.crab_speed * world_direction)
        cmd = Twist()
        cmd.linear.y = robot_vy
        self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # RETURN_CORRIDOR_ALIGN: 크랩(y) 기반 통로 중앙 보정
    # 전진(x) 없음 → 헤드랜드 벽 충돌 방지
    # ══════════════════════════════════════════════════════
    def _do_return_corridor_align(self):
        lateral_error  = self.row_state[0]
        heading_error  = self.row_state[1]
        corridor_width = self.row_state[2]

        # 통로 감지 안 되면 대기
        if corridor_width < self.crab_min_width:
            self.stop_robot()
            self.get_logger().info('⏳ 통로 감지 대기...', throttle_duration_sec=0.5)
            return

        # yaw=180도이면 lateral_error 부호 반전
        if abs(self.robot_yaw) > math.pi / 2:
            vy = max(-0.10, min(0.10, -self.Kp * lateral_error))
        else:
            vy = max(-0.10, min(0.10,  self.Kp * lateral_error))

        h = heading_error
        if abs(h) > math.pi / 2:
            h -= math.copysign(math.pi, h)
        wz = max(-0.2, min(0.2, h * 0.5))

        self.get_logger().info(
            f'[RETURN_CORRIDOR_ALIGN] lat={lateral_error:.3f}m vy={vy:.3f} '
            f'width={corridor_width:.2f}m '
            f'({self.return_corridor_align_count}/{self.RETURN_CORRIDOR_ALIGN_CONFIRM})',
            throttle_duration_sec=0.3)

        if abs(lateral_error) < self.RETURN_CORRIDOR_ALIGN_THRESH:
            self.return_corridor_align_count += 1
            if self.return_corridor_align_count >= self.RETURN_CORRIDOR_ALIGN_CONFIRM:
                self.stop_robot()
                self.get_logger().info(
                    f'✅ 통로 중앙 보정 완료 | lat={lateral_error:.3f}m → RETURN_CRAB')
                self.state = State.RETURN_CRAB
                return
        else:
            self.return_corridor_align_count = 0

        # 전진 없이 크랩(y)으로만 보정
        cmd = Twist()
        cmd.linear.y  = vy
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # RETURN_ALIGN
    # ══════════════════════════════════════════════════════
    def _do_return_align(self):
        dx = self.spawn_odom_x - self.odom_x
        target_yaw = math.pi if dx < 0 else 0.0
        yaw_error = math.atan2(
            math.sin(target_yaw - self.robot_yaw),
            math.cos(target_yaw - self.robot_yaw))

        self.get_logger().info(
            f'[RETURN_ALIGN] yaw={math.degrees(self.robot_yaw):.1f}° '
            f'target={math.degrees(target_yaw):.1f}° err={math.degrees(yaw_error):.1f}°',
            throttle_duration_sec=0.3)

        if abs(yaw_error) < math.radians(2.0):
            self.return_align_count += 1
            if self.return_align_count >= self.RETURN_ALIGN_CONFIRM:
                self.stop_robot()
                self.state = State.RETURN_FWD
                self.get_logger().info(
                    f'✅ 귀환 방향 정렬 완료 → RETURN_FWD '
                    f'(target_yaw={math.degrees(target_yaw):.1f}°)')
        else:
            self.return_align_count = 0
            wz = math.copysign(0.3, yaw_error)
            cmd = Twist()
            cmd.angular.z = wz
            self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # RETURN_FWD
    # ══════════════════════════════════════════════════════
    def _do_return_fwd(self):
        dx = self.spawn_odom_x - self.odom_x
        remaining = abs(dx)
        lateral_error = self.row_state[0]
        heading_error = self.row_state[1]
        corridor_width = self.row_state[2]

        if abs(self.robot_yaw) > math.pi / 2:
            vy = max(-0.05, min(0.05, -self.Kp * lateral_error))
        else:
            vy = max(-0.05, min(0.05,  self.Kp * lateral_error))

        h = heading_error
        if abs(h) > math.pi / 2:
            h -= math.copysign(math.pi, h)
        wz = max(-0.2, min(0.2, h * 0.5))

        self.get_logger().info(
            f'[RETURN_FWD] odom_x={self.odom_x:.2f}m 남은={remaining:.2f}m '
            f'lat={lateral_error:.3f}m vy={vy:.3f} '
            f'head={math.degrees(h):.1f}° wz={wz:.3f} width={corridor_width:.2f}m',
            throttle_duration_sec=0.3)

        if remaining <= 0.1:
            self.stop_robot()
            self.state = State.MISSION_DONE
            self.get_logger().info(f'🎉 미션 완료! 총 {self.detected_max_row}행 완주 + 귀환 완료!')
            return

        cmd = Twist()
        cmd.linear.x  = self.linear_vel
        cmd.linear.y  = vy
        cmd.angular.z = wz
        self.cmd_pub.publish(cmd)

    # ══════════════════════════════════════════════════════
    # RETURN_TURN (예비)
    # ══════════════════════════════════════════════════════
    def _do_return_turn(self):
        yaw_error = math.atan2(
            math.sin(self.pivot_target_yaw - self.robot_yaw),
            math.cos(self.pivot_target_yaw - self.robot_yaw))
        if abs(yaw_error) < math.radians(1.0):
            self.stop_robot()
            self.state = State.RETURN_CRAB
            return
        cmd = Twist()
        cmd.angular.z = math.copysign(0.4, yaw_error)
        self.cmd_pub.publish(cmd)


def main(args=None):
    rclpy.init(args=args)
    node = RowFollower()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.stop_robot()
    finally:
        node.destroy_node()
        rclpy.shutdown()


if __name__ == '__main__':
    main()