"""
    安装步骤 ：
            1、nvidia-smi  查看cuda版本
            2、https://pytorch.org/ 下载相应的pytorch对应的cuda版本  cuda最好大于等于电脑cuda版本
            3、pip install ultralytics -i https://pypi.tuna.tsinghua.edu.cn/simple
"""

import cv2

from ultralytics import YOLO

# 加载YOLOv8模型
model = YOLO("yolo11n.pt")
# model = YOLO("yolov8n-seg.pt")  #分割
# model = YOLO("yolov8n-pose.pt")  #姿势
# model = YOLO("yolov8n-obb.pt")  #旋转框

# 打开视频文件
video_path = "1.mp4"
cap = cv2.VideoCapture(video_path)  # 从视频文件获取
# cap = cv2.VideoCapture("https://upos-sz-estgoss.bilivideo.com/upgcxcode/84/18/31754291884/31754291884-1-192.mp4?e=ig8euxZM2rNcNbRBhzdVhwdlhWUzhwdVhoNvNC8BqJIzNbfq9rVEuxTEnE8L5F6VnEsSTx0vkX8fqJeYTj_lta53NCM=&oi=2018263300&deadline=1755780505&nbs=1&gen=playurlv3&os=estgoss&og=ali&platform=html5&uipk=5&mid=3546867474368918&trid=c30e53dc9d874f2486bebb14970f2f2T&upsig=84079789a7344d552de1818be7342256&uparams=e,oi,deadline,nbs,gen,os,og,platform,uipk,mid,trid&bvc=vod&nettype=0&bw=1233744&build=0&dl=0&f=T_0_0&mobi_app=&agrr=1&buvid=&orderid=0,1")     # 从摄像头获取
# cap = cv2.VideoCapture(0)  # 从摄像头获取


# 遍历视频帧
while cap.isOpened():
    # 从视频中读取一帧
    success, frame = cap.read()

    if success:
        # 在帧上运行YOLOv8推理
        results = model(frame)

        # 在帧上可视化推理结果
        annotated_frame = results[0].plot()

        # 显示标注后的帧
        cv2.imshow("YOLOv8推理结果", annotated_frame)

        # 如果按下'q'键则退出循环
        if cv2.waitKey(1) & 0xFF == ord("q"):
            break
    else:
        # 如果视频播放完毕，则退出循环
        break

# 释放视频捕获对象并关闭显示窗口
cap.release()
cv2.destroyAllWindows()