from hqyj_mqtt import Mqtt_Clt  # 导入自定义的MQTT客户端模块
import os  # 导入操作系统接口模块
import base64
import numpy as np
import cv2

def GetCvImage(jsonMsg,key):
    # 将Base64编码解码为原始的二进制数据
    image_data = base64.b64decode(jsonMsg[key])
    # 将二进制数据转换为一个np.uint8类型的numpy数组
    image_array = np.frombuffer(image_data, np.uint8)
    # 将numpy数组转换为opencv图像对象
    image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
    return image


# 定义创建目录的函数
def make_dir(folder_path):
    if not os.path.exists(folder_path):  # 检查目录是否存在
        print(f'{folder_path}不存在')  # 输出目录不存在的信息
        os.makedirs(folder_path)  # 创建目录
        print(f'{folder_path}已创建')  # 输出目录已创建的信息
    else:
        print(f'{folder_path}已存在，无需创建')  # 输出目录已存在，无需创建的信息

# 初始化MQTT客户端
mqtt_client = Mqtt_Clt('127.0.0.1', 21883, 'bb', 'aa', 60)

# 设置存储图片的目录路径
folder_path = './data'
make_dir(folder_path)  # 创建目录

# 初始化图片编号
i = 0

# 无限循环，等待接收MQTT消息
while True:
    # 从MQTT队列中获取消息
    json_msg = mqtt_client.mqtt_queue.get()
    # 检查消息中是否包含'image'键
    if 'image' in json_msg:
        print(json_msg)  # 打印接收到的消息
        # 使用GetCvImage函数将消息中的图像数据转换为OpenCV图像格式
        image = GetCvImage(json_msg, 'image')
        # 将图像保存到指定目录
        cv2.imwrite(folder_path + '/' + f'img{i}.jpg', image)
        print(f'图片{i}已保存')  # 输出图片保存成功的信息
        i += 1  # 图片编号递增