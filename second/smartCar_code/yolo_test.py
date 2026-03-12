

from tools import *


if __name__ == '__main__':

    # 在帧上运行YOLOv8推理  9为traffic light
    """
        类型 ： result.boxes.cls.tolist()
        中心点坐标：xywh = result.boxes.xywh.tolist()  
    """
    # 通过opencv绘制矩形
    image = cv2.imread('./ren3.png')
    light_color = yolo_detect(image)
    if light_color is not None:
        print('识别到的颜色为:',light_color)
    cv2.waitKey()
