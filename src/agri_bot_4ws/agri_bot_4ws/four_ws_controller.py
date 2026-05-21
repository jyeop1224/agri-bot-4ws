import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math

class FourWSController(Node):
    def __init__(self):
        super().__init__('four_ws_controller')
        self.declare_parameter('wheel_base',   0.4)
        self.declare_parameter('track_width',  0.44)
        self.declare_parameter('wheel_radius', 0.1)
        self.declare_parameter('max_steer',    1.5707)
        self.L = self.get_parameter('wheel_base').value
        self.W = self.get_parameter('track_width').value
        self.R = self.get_parameter('wheel_radius').value
        self.max_steer = self.get_parameter('max_steer').value

        self.current_steer = [0.0] * 4
        self.steer_names = [
            'front_left_steering_joint',
            'front_right_steering_joint',
            'rear_left_steering_joint',
            'rear_right_steering_joint',
        ]

        self.sub_cmd = self.create_subscription(
            Twist, '/cmd_vel', self.cmd_vel_callback, 10)
        self.sub_js = self.create_subscription(
            JointState, '/joint_states', self.joint_cb, 10)
        self.pub_steer = self.create_publisher(
            Float64MultiArray, '/steering_controller/commands', 10)
        self.pub_wheel = self.create_publisher(
            Float64MultiArray, '/wheel_controller/commands', 10)
        self.get_logger().info('4WS Controller 시작')
        self.get_logger().info(f'wheel_base={self.L}, track_width={self.W}')

    def joint_cb(self, msg):
        name_idx = {n: i for i, n in enumerate(msg.name)}
        for i, n in enumerate(self.steer_names):
            if n in name_idx:
                idx = name_idx[n]
                if idx < len(msg.position):
                    self.current_steer[i] = msg.position[idx]

    def cmd_vel_callback(self, msg: Twist):
        vx = msg.linear.x
        vy = msg.linear.y
        wz = msg.angular.z
        THRESH = 0.01

        if abs(vy) > THRESH and abs(vx) < THRESH and abs(wz) < THRESH:
            steer_fl, steer_fr, steer_rl, steer_rr, \
            vel_fl, vel_fr, vel_rl, vel_rr = self.crab_mode(vy)
            mode = 'CRAB'
        elif abs(wz) > THRESH and abs(vx) < THRESH and abs(vy) < THRESH:
            steer_fl, steer_fr, steer_rl, steer_rr, \
            vel_fl, vel_fr, vel_rl, vel_rr = self.pivot_mode(wz)
            mode = 'PIVOT'
        elif abs(vx) > THRESH:
            steer_fl, steer_fr, steer_rl, steer_rr, \
            vel_fl, vel_fr, vel_rl, vel_rr = self.ackermann_mode(vx, wz)
            mode = 'ACKERMANN'
        else:
            steer_fl = steer_fr = steer_rl = steer_rr = 0.0
            vel_fl = vel_fr = vel_rl = vel_rr = 0.0
            mode = 'STOP'

        steer_fl = self.clamp(steer_fl, -self.max_steer, self.max_steer)
        steer_fr = self.clamp(steer_fr, -self.max_steer, self.max_steer)
        steer_rl = self.clamp(steer_rl, -self.max_steer, self.max_steer)
        steer_rr = self.clamp(steer_rr, -self.max_steer, self.max_steer)

        steer_msg = Float64MultiArray()
        steer_msg.data = [steer_fl, steer_fr, steer_rl, steer_rr]
        self.pub_steer.publish(steer_msg)

        wheel_msg = Float64MultiArray()
        wheel_msg.data = [vel_fl, vel_fr, vel_rl, vel_rr]
        self.pub_wheel.publish(wheel_msg)

        self.get_logger().info(
            f'[{mode}] steer=[{steer_fl:.2f},{steer_fr:.2f},'
            f'{steer_rl:.2f},{steer_rr:.2f}] '
            f'vel=[{vel_fl:.2f},{vel_fr:.2f},{vel_rl:.2f},{vel_rr:.2f}]',
            throttle_duration_sec=0.5
        )

    def ackermann_mode(self, vx, wz):
        if abs(wz) < 1e-6:
            vel = vx / self.R
            return 0.0, 0.0, 0.0, 0.0, vel, vel, vel, vel
        R_turn = abs(vx / wz)
        sign = 1.0 if wz > 0 else -1.0
        R_inner = R_turn - self.W / 2
        R_outer = R_turn + self.W / 2
        delta_inner = math.atan(self.L/2 / R_inner) if abs(R_inner) > 0.01 else self.max_steer
        delta_outer = math.atan(self.L/2 / R_outer) if abs(R_outer) > 0.01 else self.max_steer
        steer_fl = sign * delta_inner
        steer_fr = sign * delta_outer
        steer_rl = -sign * delta_inner
        steer_rr = -sign * delta_outer
        base = abs(vx) / self.R
        dir_sign = 1.0 if vx > 0 else -1.0
        vel_fl = dir_sign * base * R_inner / R_turn if R_turn > 0.01 else base
        vel_fr = dir_sign * base * R_outer / R_turn if R_turn > 0.01 else base
        vel_rl = vel_fl
        vel_rr = vel_fr
        return steer_fl, steer_fr, steer_rl, steer_rr, vel_fl, vel_fr, vel_rl, vel_rr

    def crab_mode(self, vy):
        # steering 고정
        steer = math.pi / 2

        # velocity 부호만 사용
        vel = vy / self.R
        return steer, steer, steer, steer, vel, vel, vel, vel

    def pivot_mode(self, wz):
        alpha = math.atan2(self.L / 2, self.W / 2)
        r = math.hypot(self.L / 2, self.W / 2)
        vel = abs(wz) * r / self.R
        sign = 1.0 if wz > 0 else -1.0

        # 조향각은 j/l 모두 동일 (마름모 고정)
        steer_fl = -alpha
        steer_fr = +alpha
        steer_rl = +alpha
        steer_rr = -alpha

        # 속도 부호만 바꿔서 회전 방향 결정
        vel_fl = -sign * vel
        vel_fr = +sign * vel
        vel_rl = -sign * vel
        vel_rr = +sign * vel

        return steer_fl, steer_fr, steer_rl, steer_rr, \
            vel_fl, vel_fr, vel_rl, vel_rr

    def clamp(self, val, mn, mx):
        return max(mn, min(mx, val))

def main(args=None):
    rclpy.init(args=args)
    node = FourWSController()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
