import threading
from time import sleep
import numpy as np
import cv2
from ultralytics import YOLO
from libraries.dobot_api import DobotApiDashboard, DobotApi, DobotApiMove, MyType
import time

# 전역 변수 (현재 좌표)
current_actual = None

# 입력 파라미터
ip = "192.168.1.6"
gripper_port = 1
speed_value = 100

# Robot의 IP 주소
# 그리퍼 포트 번호
# 로봇 속도 (1~100 사이의 값 입력)
# 로봇이 이동하고자 하는 좌표 (x, y, z, yaw) unit : mm, degree
point_home = [168.170372,-202.957353, 50, 115]

max_x, min_x = 380, 190
max_y, min_y = 130, -130


block_gap = 16.5
min_z = -58

# ARUCO ID
# 1 - red, 2 - green, 3 - yellow
# classes.txt
# 0 - red, 1 - green, 2 - yellow


# status variables
marker_points = [[246.150598,-118.756685], [336.290271,-7.042964], [254.350688,100.121519]] # tvec -> order : red, green, yellow
hanoi_stack = [] # values : [x, y]
hanoi_cls_stack = []
base_point = None # value : [x, y]
top_color = None


model = None
cam = None
frame = None
cap_cnt = 0


### main ###

def main():
    global base_point, cam, model, frame, top_color


    model = YOLO("best.pt")

    # cam = cv2.VideoCapture(1)

    try:
        # 로봇 연결
        dashboard, move, feed = connect_robot(ip)
        dashboard.EnableRobot()
        # print("이제 로봇을 사용할 수 있습니다!")
        time.sleep(0.1)

        # 쓰레드 설정
        feed_thread = threading.Thread(target=get_feed, args=(feed,))
        feed_thread.setDaemon(True)
        feed_thread.start()

        # cam_thread = threading.Thread(target=capture_frame, args=())
        # cam_thread.setDaemon(True)
        # cam_thread.start()
        
        # 로봇 상태 초기화 1 : 로봇 에러 메시지 초기화
        robot_clear(dashboard)
        # 로봇 상태 초기화 2 : 로봇 속도 조절
        robot_speed(dashboard, speed_value)
        # 로봇 현재 위치 받아오기 (x, y, z, yaw) - 로봇 베이스 좌표계
        get_Pose(dashboard)

        # 로봇 구동 1 (Home)
        run_point(move, point_home)
        wait_arrive(point_home)
        # cam = cv2.VideoCapture("20231204-113742.jpg")

        # aruco_pose_esitmation(frame)
        print(f"Aruco Point : {marker_points}")

        # base_point = marker_points[0]
        # tmp = [0, 1, 2]


        flag = True
        # capture_frame()
        cam = cv2.VideoCapture(1)
        ret, frame = cam.read()
        # cv2.imwrite("capture.jpg", frame)
        cam.release()
        time.sleep(0.1)
        find_base_point(flag)   
        hanoi_cls_stack.append(top_color)
        hanoi_stack.append(marker_points[top_color])

        for i in range(3):
            flag = False

            # base_point = [300, -112]
            print(f"Base Point : {base_point}")
            print(f"top_color: {top_color}")

            target = base_point + [(min_z + block_gap * (2-i)), 180]
            print(target)
            run_point(move, target)
            wait_arrive(target)

            # 그리퍼 구동
            gripper_DO(dashboard, gripper_port, 1)
            sleep(0.1)

            target = base_point + [50, 180]
            run_point(move, target)
            wait_arrive(target)


            target = marker_points[top_color] + [min_z, 180]
            run_point(move, target)
            wait_arrive(target)

            if i != 2:
                cam = cv2.VideoCapture(1)
                ret, frame = cam.read()
                # cv2.imwrite("capture.jpg", frame)
                cam.release()
                time.sleep(0.3)
                find_base_point(flag)
                hanoi_cls_stack.append(top_color)
                hanoi_stack.append(marker_points[top_color])

                target = base_point + [0, 180]
                run_point(move, target)
                wait_arrive(target)

            # 그리퍼 끄기
            gripper_DO(dashboard, gripper_port, 0)
            sleep(0.1)

            run_point(move, point_home)
            wait_arrive(point_home)

        print(hanoi_stack)
        base_point = hanoi_stack[-1]

        for i in range(3):
            target = hanoi_stack.pop() + [min_z, 180]

            if i==0:
                continue

            run_point(move, target)
            wait_arrive(target)

            # 그리퍼 구동
            gripper_DO(dashboard, gripper_port, 1)
            sleep(0.1)

            target = base_point + [0, 180]
            run_point(move, target)
            wait_arrive(target)

            target = base_point + [(min_z + block_gap * (i+1)), 180]
            run_point(move, target)
            wait_arrive(target)

            # 그리퍼 끄기
            gripper_DO(dashboard, gripper_port, 0)
            sleep(0.1)

            target = base_point + [0 - block_gap * (2-i) + block_gap, 180]
            run_point(move, target)
            wait_arrive(target)


        # 로봇 구동 1 (Home)
        run_point(move, point_home)
        wait_arrive(point_home)
    except:
        print("error occur")
    finally:
        # 로봇 끄기
        dashboard.DisableRobot()

def find_base_point(flag):
    global frame, model, base_point, top_color, hanoi_cls_stack, cap_cnt
    # if cap_cnt >= 4:
    #     return
    
    # frame = cv2.imread("capture.jpg")

    result = model.predict(frame)
    print(result[0])
    
    box = result[0].boxes

    i = 0
    next_color = int(box.cls[0].cpu())
    print(next_color, hanoi_cls_stack, len(box.cls))
    while next_color in hanoi_cls_stack:
        i += 1
        next_color = int(box.cls[i].cpu())
    top_color = next_color

    if flag:
        xywh = box.xywh[0].tolist()
        x, y = xywh[0], xywh[1]
        base_point = transform_cam_pixel_to_mm(x, y)

# 변환 행렬 코드

def transform_cam_pixel_to_mm(px, py):
    rx = 166.85991724  + 0.4564514 * py + px * 0.02
    ry = -147.63000263 + px * 0.45 - 0.02903971 * py

    return [rx, ry]


### ROBOT FUNCTIONS ###

def connect_robot(ip):
    try:
        dashboard_p = 29999
        move_p = 30003
        feed_p = 30004
        print("연결 설정 중...")
        dashboard = DobotApiDashboard(ip, dashboard_p)
        move = DobotApiMove(ip, move_p)
        feed = DobotApi(ip, feed_p)
        print("연결 성공!!")
        return dashboard, move, feed
    
    except Exception as e:
        print("연결 실패")
        raise e
    
def robot_clear(dashboard : DobotApiDashboard):
    dashboard.ClearError()

def robot_speed(dashboard : DobotApiDashboard, speed_value):
    dashboard.SpeedFactor(speed_value)

def gripper_DO(dashboard : DobotApiDashboard, index, status):
    dashboard.ToolDO(index, status)

def get_Pose(dashboard : DobotApiDashboard):
    dashboard.GetPose()

def run_point(move: DobotApiMove, point_list: list):
    move.MovL(point_list[0], point_list[1], point_list[2], point_list[3])

def get_feed(feed: DobotApi):
    global current_actual
    hasRead = 0

    while True:
        data = bytes()

        while hasRead < 1440:
            temp = feed.socket_dobot.recv(1440 - hasRead)
            if len(temp) > 0:
                hasRead += len(temp)
                data += temp

        hasRead = 0
        a = np.frombuffer(data, dtype=MyType)
        
        if hex((a['test_value'][0])) == '0x123456789abcdef':
            current_actual = a["tool_vector_actual"][0] # Refresh Properties
        sleep(0.001)

def wait_arrive(point_list):
    global current_actual
    while True:
        is_arrive = True
        if current_actual is not None:
            for index in range(4):
                if (abs(current_actual[index] - point_list[index]) > 1):
                    is_arrive = False
        if is_arrive:
            return
        sleep(0.001)

if __name__ == "__main__":
    main()