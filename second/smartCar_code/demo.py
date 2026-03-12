import time

import cv2

from hqyj_mqtt import Mqtt_Clt
from lineDetect import *
import base64

from pid import PID
from tools import *
def auto_run(image,mqtt_client,pts_middle,pid,carspeed = 20):
    if pts_middle is None:
        return
    # 得到车的当前位置x的值，位置为常量
    image_center= image.shape[1]//2
    # 控制速度
    mqtt_client.control_device('carSpeed',carspeed)
    # 处理目标位置 得到车道线的x值  pts_middle(x,y)这种排列的坐标点
    land_center = pts_middle[240:,::].mean(axis=0)[0]
    # print(land_center)

    steering_angle =  -pid(land_center)
    mqtt_client.control_device('carDirection', steering_angle)

    # print('steering_angle:',steering_angle)

    return land_center,image_center

if __name__ == '__main__':
    global stop_line_flag
    ip_broker = '127.0.0.1'
    port_broker = 21883
    topic_sub = 'bb'
    topic_pub = 'aa'
    time_out_secs = 60

    pid = PID(Kp=0.2, Ki=0.0, Kd=0.0, setpoint=240)
    pid.sample_time = 0.1
    pid.output_limits = (-13, 13)

    client = Mqtt_Clt(ip_broker, port_broker, topic_sub, topic_pub, time_out_secs)

    while True:
        # 接收数据

        json_msg = client.mqtt_queue.get()

        # 判断是否为图像数据
        if 'image' in json_msg:
            # 将base64编码解码为原始的二进制数据
            image_data = base64.b64decode(json_msg['image'])
            # 将二进制数据转换为np.uint8类型的数组
            image_array = np.frombuffer(image_data, np.uint8)
            # 将numpy数组转换为opencv图像  实际上是bgr
            image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

            # cv2.imwrite('ren3.png', image_rgb)

            # 透视变换
            image_arpPerspective,M_INV = perspective_transform(image_bgr)
            # 图像增强
            img_close = image_enhance(image_arpPerspective.copy())
            stop_line_flag = find_stop_line(image_arpPerspective.copy())

            light_color,person_Warning = yolo_detect(image_bgr)
            if  person_Warning == True:
                print('前方检测到行人，，需要刹车！！！')
                client.control_device('carSpeed', 0)
                time.sleep(1)
                continue
            if stop_line_flag==True:  # 停止的状态
                client.control_device('carSpeed', 0)
                # time.sleep(1)

                print('识别到的颜色为:', light_color)
                if light_color != None:
                    if light_color == 0 or light_color == 1:

                        print('停止车辆')
                        client.control_device('carSpeed', 0)
                        cv2.waitKey(1)
                        continue
                    elif light_color==2:
                        print('绿灯，启动车辆')
                        client.control_device('carSpeed', 20)
                        time.sleep(2)

            lineLoc = find_line(img_close)
            pts_middle = show_line(image_bgr, image_arpPerspective, lineLoc, M_INV)

            auto_run(image_bgr,client,pts_middle,pid)



            cv2.waitKey(1)