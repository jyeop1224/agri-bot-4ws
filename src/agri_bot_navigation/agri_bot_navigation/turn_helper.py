import rclpy
from rclpy.node import Node
from geometry_msgs.msg import Twist
import time

def main():
    rclpy.init()
    node = Node('turn_helper')
    pub = node.create_publisher(Twist, '/cmd_vel', 10)
    time.sleep(0.5)

    print('=== 회전 헬퍼 ===')
    print('1: 우회전 90도 (아커만)')
    print('2: 좌회전 90도 (아커만)')
    print('q: 종료')

    while True:
        cmd = input('선택: ')

        if cmd == '1':
            print('우회전 90도 실행...')
            # 아커만 우회전
            msg = Twist()
            msg.linear.x = 0.15
            msg.angular.z = -0.6
            end = time.time() + 2.6  # 90도
            while time.time() < end:
                pub.publish(msg)
                time.sleep(0.05)
            # 정지
            pub.publish(Twist())
            print('완료')

        elif cmd == '2':
            print('좌회전 90도 실행...')
            msg = Twist()
            msg.linear.x = 0.15
            msg.angular.z = 0.6
            end = time.time() + 2.6
            while time.time() < end:
                pub.publish(msg)
                time.sleep(0.05)
            pub.publish(Twist())
            print('완료')

        elif cmd == 'q':
            break

    rclpy.shutdown()

if __name__ == '__main__':
    main()
