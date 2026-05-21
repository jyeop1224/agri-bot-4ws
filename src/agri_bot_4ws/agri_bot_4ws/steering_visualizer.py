import rclpy
from rclpy.node import Node
from std_msgs.msg import Float64MultiArray
from sensor_msgs.msg import JointState
import math

class SteeringVisualizer(Node):
    """
    /steering_angles 수신 → /joint_states 발행
    Gazebo + RViz2에서 조향 시각화
    """
    def __init__(self):
        super().__init__('steering_visualizer')

        self.joint_names = [
            'front_left_steering_joint',
            'front_right_steering_joint',
            'rear_left_steering_joint',
            'rear_right_steering_joint',
            'front_left_wheel_joint',
            'front_right_wheel_joint',
            'rear_left_wheel_joint',
            'rear_right_wheel_joint',
        ]

        self.steer_angles = [0.0] * 4
        self.wheel_vels   = [0.0] * 4
        self.wheel_pos    = [0.0] * 4

        self.sub_steer = self.create_subscription(
            Float64MultiArray, '/steering_angles',
            self.steer_cb, 10)

        self.sub_wheel = self.create_subscription(
            Float64MultiArray, '/wheel_velocities',
            self.wheel_cb, 10)

        self.pub_joint = self.create_publisher(
            JointState, '/joint_states', 10)

        self.create_timer(0.05, self.publish_joints)  # 20Hz
        self.get_logger().info('Steering Visualizer 시작')

    def steer_cb(self, msg):
        if len(msg.data) == 4:
            self.steer_angles = list(msg.data)

    def wheel_cb(self, msg):
        if len(msg.data) == 4:
            self.wheel_vels = list(msg.data)

    def publish_joints(self):
        msg = JointState()
        msg.header.stamp = self.get_clock().now().to_msg()
        msg.name     = self.joint_names
        msg.position = self.steer_angles + self.wheel_pos
        msg.velocity = [0.0] * 4 + self.wheel_vels

        # 바퀴 위치 적분
        dt = 0.05
        for i in range(4):
            self.wheel_pos[i] += self.wheel_vels[i] * dt

        self.pub_joint.publish(msg)

def main(args=None):
    rclpy.init(args=args)
    node = SteeringVisualizer()
    rclpy.spin(node)
    node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
