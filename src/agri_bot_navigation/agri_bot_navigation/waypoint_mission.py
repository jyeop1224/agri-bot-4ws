import rclpy
from nav2_simple_commander.robot_navigator import BasicNavigator, TaskResult
from geometry_msgs.msg import PoseStamped
import tf_transformations
import time

def make_pose(nav, x, y, yaw=0.0):
    pose = PoseStamped()
    pose.header.frame_id = 'map'
    pose.header.stamp = nav.get_clock().now().to_msg()
    pose.pose.position.x = x
    pose.pose.position.y = y
    q = tf_transformations.quaternion_from_euler(0, 0, yaw)
    pose.pose.orientation.x = q[0]
    pose.pose.orientation.y = q[1]
    pose.pose.orientation.z = q[2]
    pose.pose.orientation.w = q[3]
    return pose

def main():
    rclpy.init()
    nav = BasicNavigator()

    print('Nav2 활성화 대기 중...')
    nav.waitUntilNav2Active()
    print('Nav2 준비 완료. 미션 시작.')

    # 현재 위치: x=-0.977, y=-2.232 기준
    # 온실 작물열 패턴으로 4개 지점 순회 후 복귀
    waypoints = [
        make_pose(nav, -0.977,  -1.0,  1.57),  # 1번: 북쪽으로 전진
        make_pose(nav, -0.977,   0.0,  1.57),  # 2번: 계속 북쪽
        make_pose(nav,  0.5,     0.0,  0.0),   # 3번: 동쪽으로 이동
        make_pose(nav,  0.5,    -1.0, -1.57),  # 4번: 남쪽으로 이동
        make_pose(nav, -0.977,  -2.232, 3.14), # 5번: 홈 복귀
    ]

    total = len(waypoints)
    print(f'총 {total}개 Waypoint 순회 시작')
    print('경로: 현재위치 → 북쪽 이동 → 동쪽 이동 → 남쪽 이동 → 홈 복귀')

    nav.followWaypoints(waypoints)

    while not nav.isTaskComplete():
        feedback = nav.getFeedback()
        if feedback:
            current = feedback.current_waypoint + 1
            print(f'  → {current} / {total} 번째 Waypoint 이동 중...')
        time.sleep(1.0)

    result = nav.getResult()

    if result == TaskResult.SUCCEEDED:
        print('✓ 미션 완료: 모든 Waypoint 순회 성공!')
    elif result == TaskResult.CANCELED:
        print('✗ 미션 취소됨')
    elif result == TaskResult.FAILED:
        print('✗ 미션 실패: 경로를 찾지 못했습니다')

    nav.lifecycleShutdown()
    rclpy.shutdown()

if __name__ == '__main__':
    main()
