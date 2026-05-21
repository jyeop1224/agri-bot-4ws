import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from nav_msgs.msg import Odometry
from geometry_msgs.msg import TransformStamped, Twist
from tf2_ros import TransformBroadcaster
import math

class OdometryPublisher(Node):
    def __init__(self):
        super().__init__('odometry_publisher')
        self.L = 0.4
        self.W = 0.44
        self.R = 0.1
        self.x = 0.0
        self.y = 0.0
        self.yaw = 0.0
        self.last_time = self.get_clock().now()
        self.wheel_names = [
            'front_left_wheel_joint',
            'front_right_wheel_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint',
        ]
        self.steer_names = [
            'front_left_steering_joint',
            'front_right_steering_joint',
            'rear_left_steering_joint',
            'rear_right_steering_joint',
        ]
        self.wheel_vel = [0.0] * 4
        self.steer_ang = [0.0] * 4
        self.cmd_vx = 0.0
        self.cmd_wz = 0.0

        self.sub_js = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10)
        self.sub_cmd = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_cb, 10)
        self.pub_odom = self.create_publisher(Odometry, '/odom', 10)
        self.tf_br = TransformBroadcaster(self)
        self.create_timer(0.05, self.publish_odom)
        self.get_logger().info('Odometry Publisher 시작')

    def cmd_cb(self, msg):
        self.cmd_vx = msg.linear.x
        self.cmd_wz = msg.angular.z

    def joint_cb(self, msg):
        name_idx = {n: i for i, n in enumerate(msg.name)}
        new_vel = [0.0] * 4
        valid = True
        for i, n in enumerate(self.wheel_names):
            if n in name_idx:
                idx = name_idx[n]
                if idx < len(msg.velocity):
                    new_vel[i] = msg.velocity[idx]
            else:
                valid = False
        if valid:
            self.wheel_vel = new_vel
        for i, n in enumerate(self.steer_names):
            if n in name_idx:
                idx = name_idx[n]
                if idx < len(msg.position):
                    self.steer_ang[i] = msg.position[idx]

    def publish_odom(self):
        try:
            now = self.get_clock().now()
            dt = (now - self.last_time).nanoseconds / 1e9
            self.last_time = now
            if dt <= 0 or dt > 1.0:
                return

            # 실제 바퀴 평균 선속도
            v_avg = sum(v * self.R for v in self.wheel_vel) / 4.0

            # 피벗 반경: 로봇 중심 → 바퀴까지 거리
            pivot_r = math.sqrt((self.L / 2) ** 2 + (self.W / 2) ** 2)

            # 피벗 모드: FL, RR = 양수, FR, RL = 음수 (또는 반대)
            # 좌측 바퀴: FL(0), RL(2) / 우측 바퀴: FR(1), RR(3)
            v_left  = (self.wheel_vel[0] + self.wheel_vel[2]) / 2.0  # FL, RL
            v_right = (self.wheel_vel[1] + self.wheel_vel[3]) / 2.0  # FR, RR

            # encoder 기반 wz 계산
            # 피벗 모드: 좌우 바퀴가 반대 방향
            # 일반 주행: 좌우 속도 차이로 wz 계산
            if abs(self.cmd_vx) < 0.01 and abs(self.cmd_wz) > 0.01:
                v_left_ms  = v_left * self.R
                v_right_ms = v_right * self.R
                wz = (v_right_ms - v_left_ms) / (2.0 * pivot_r)
            elif abs(self.cmd_vx) > 0.01 and abs(v_avg) > 0.001:
                wz = (v_right - v_left) * self.R / self.W
            else:
                wz = 0.0

            # 앞바퀴 평균 조향각
            steer_front = (self.steer_ang[0] + self.steer_ang[1]) / 2.0

            # 로컬 속도
            if abs(self.cmd_vx) > 0.01 and abs(steer_front) < 0.17:
                vx_local = v_avg
                vy_local = 0.0
            else:
                vx_local = v_avg * math.cos(steer_front)
                vy_local = v_avg * math.sin(steer_front)

            # 글로벌 좌표 변환
            vx = vx_local * math.cos(self.yaw) - vy_local * math.sin(self.yaw)
            vy = vx_local * math.sin(self.yaw) + vy_local * math.cos(self.yaw)

            # 위치 적분
            self.x   += vx * dt
            self.y   += vy * dt
            self.yaw += wz * dt

            q = [0.0, 0.0, math.sin(self.yaw/2), math.cos(self.yaw/2)]

            # Odometry 발행
            odom = Odometry()
            odom.header.stamp = now.to_msg()
            odom.header.frame_id = 'odom'
            odom.child_frame_id = 'base_footprint'
            odom.pose.pose.position.x = self.x
            odom.pose.pose.position.y = self.y
            odom.pose.pose.orientation.x = q[0]
            odom.pose.pose.orientation.y = q[1]
            odom.pose.pose.orientation.z = q[2]
            odom.pose.pose.orientation.w = q[3]
            odom.twist.twist.linear.x  = vx_local
            odom.twist.twist.linear.y  = vy_local
            odom.twist.twist.angular.z = wz

            # pose covariance [x, y, z, roll, pitch, yaw] 6x6 대각
            odom.pose.covariance[0]  = 0.05   # x
            odom.pose.covariance[7]  = 0.05   # y
            odom.pose.covariance[14] = 1e-4   # z (평면 이동)
            odom.pose.covariance[21] = 1e-4   # roll
            odom.pose.covariance[28] = 1e-4   # pitch
            odom.pose.covariance[35] = 0.1    # yaw

            # twist covariance
            odom.twist.covariance[0]  = 0.05  # vx
            odom.twist.covariance[7]  = 0.05  # vy
            odom.twist.covariance[35] = 0.1   # wz

            self.pub_odom.publish(odom)

            # TF 발행
            tf = TransformStamped()
            tf.header.stamp = now.to_msg()
            tf.header.frame_id = 'odom'
            tf.child_frame_id = 'base_footprint'
            tf.transform.translation.x = self.x
            tf.transform.translation.y = self.y
            tf.transform.translation.z = 0.0
            tf.transform.rotation.x = q[0]
            tf.transform.rotation.y = q[1]
            tf.transform.rotation.z = q[2]
            tf.transform.rotation.w = q[3]
            self.tf_br.sendTransform(tf)

            self.get_logger().info(
                f'pos=({self.x:.2f},{self.y:.2f}) '
                f'yaw={math.degrees(self.yaw):.1f}° '
                f'v={v_avg:.3f} wz={wz:.3f}',
                throttle_duration_sec=0.5
            )
        except Exception as e:
            self.get_logger().warn(f'odom error: {e}')

def main(args=None):
    rclpy.init(args=args)
    node = OdometryPublisher()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
