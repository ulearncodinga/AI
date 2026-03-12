
import cv2
import matplotlib.pyplot as plt
import numpy as np

from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("yolo11n.pt")


def yolo_detect(image):
    results = model(image,verbose= False)
    light_color = None
    person_Warning = False
    for result in results:
        # result.show()
        cls = result.boxes.cls.tolist()
        if len(cls)>0:
            cls = cls[0]
            if cls == 9:
                # 获取置信度
                conf = result.boxes.conf.tolist()[0]
                xywh = result.boxes.xywh.tolist()[0]
                if conf > 0.5:
                 light_color =  detect_light_color(image.copy(),xywh)
            elif cls == 127:
                conf = result.boxes.conf.tolist()[0]
                xywh = result.boxes.xywh.tolist()[0]
                if conf > 0.5:
                    stop_line = detect_stop_line(image.copy(),xywh)
            elif cls == 0:
                # 获取置信度
                conf = result.boxes.conf.tolist()[0]
                xywh = result.boxes.xywh.tolist()[0]
                if conf > 0.5:
                    person_Warning = detect_person(image.copy(),xywh)


    return light_color,person_Warning

# 检测行人
def detect_person(image,xywh):
    # 创建一个变量表示前面有行人需要刹车
    person_Warning = False
    top_left = (int(xywh[0] - xywh[2] // 2), int(xywh[1] - xywh[3] // 2))
    bottom_right = (int(xywh[0] + xywh[2] // 2), int(xywh[1] + xywh[3] // 2))
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    print('前方有行人 请注意礼让行人')
    if bottom_right[1]>150:
        # print('请立即刹车')
        person_Warning = True
    return person_Warning

    # plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
    # plt.show()


def detect_stop_line(image,xywh):
    stop_line = False
    top_left = (int(xywh[0] - xywh[2] // 2), int(xywh[1] - xywh[3] // 2))
    bottom_right = (int(xywh[0] + xywh[2] // 2), int(xywh[1] + xywh[3] // 2))
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    print('检测到停止线')
    if bottom_right < 110:
        stop_line = True
    return stop_line






# 识别红绿灯颜色
def detect_light_color(image,xywh):
    """
        传入识别到的结果
    :param image:  传入原始图片 用来进行roi裁剪得出交通灯区域
    :param xywh : 交通灯所在区域的中心点坐标 以及宽度和高度
    :return:  light_color 0 红色 1 黄色 2 绿色
    """
    print('traffic light')
    # 通过中心点坐标 取一个红绿灯图
    top_left = (int(xywh[0] - xywh[2] // 2), int(xywh[1] - xywh[3] // 2))
    bottom_right = (int(xywh[0] + xywh[2] // 2), int(xywh[1] + xywh[3] // 2))
    cv2.rectangle(image, top_left, bottom_right, (255, 0, 0), 2)
    """将红绿灯切出来"""
    light_img = image[top_left[1]:bottom_right[1], top_left[0]:bottom_right[0]]
    # HSV空间转换 将RGB颜色空间下的图像转换为HSV颜色空间下的图像
    hsv_image_np = cv2.cvtColor(light_img, cv2.COLOR_BGR2HSV)
    red1_low = [0, 43, 46]
    red1_up = [10, 255, 255]
    red2_low = [156, 43, 46]
    red2_up = [180, 255, 255]

    yellow_low = [26, 43, 46]
    yellow_up = [34, 255, 255]

    green_low = [60, 50, 70]
    green_up = [70, 255, 255]
    mask1 = cv2.inRange(hsv_image_np, np.array(red1_low), np.array(red1_up))
    mask1 += cv2.inRange(hsv_image_np, np.array(red2_low), np.array(red2_up))
    # 记录红色像素点个数
    red_cnt = np.sum(mask1 == 255)
    print('红色像素点为：', red_cnt)
    # cv2.imshow('mask1', mask1)
    mask2 = cv2.inRange(hsv_image_np, np.array(yellow_low), np.array(yellow_up))
    yellow_cnt = np.sum(mask2 == 255)
    print('黄色像素点为：', yellow_cnt)
    # cv2.imshow('mask2', mask2)
    mask3 = cv2.inRange(hsv_image_np, np.array(green_low), np.array(green_up))
    green_cnt = np.sum(mask3 == 255)
    print('绿色像素点为：', green_cnt)
    # cv2.imshow('mask3', mask3)

    light_color =  np.argmax([red_cnt, yellow_cnt, green_cnt])
    # 绘制各颜色分量图片
    # plt.subplot(1, 4, 1), plt.imshow(mask1, cmap='gray')
    # plt.subplot(1, 4, 2), plt.imshow(mask2, cmap='gray')
    # plt.subplot(1, 4, 3), plt.imshow(mask3, cmap='gray')
    # plt.subplot(1, 4, 4), plt.imshow(cv2.cvtColor(light_img, cv2.COLOR_BGR2RGB))
    # plt.show()
    # plt.pause(1)

    return light_color