import mqtt
import queue
import base64
import numpy as np
import cv2



def b64_to_np(image):
    img_data = base64.b64decode(image['image'])

    #将字节数组转换为Numpy数组
    img_np = np.frombuffer(img_data,dtype=np.uint8)

    #使用opencv读取该数组
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)

    # cv2.imshow('img',img)
    # cv2.waitKey(0)

    return img


if __name__ == '__main__':
    #创建消息队列
    q_mqtt_data = queue.Queue(5)
    i = 1
    #1.构建mqtt客户端,连接mqtt服务器,方便和3D场景去通信
    mqtt_client = mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)

    while True:
        image = q_mqtt_data.get()

        if 'image' in image:
            #将接收到的消息解析为opencv处理能用的
            img = b64_to_np(image)
            cv2.imshow('img',img)
            key = cv2.waitKey(1)
            if key==ord('q'):
                break
            elif key == ord('z'):
                cv2.imwrite(f'./{i}.png', img)
                i += 1
                print('save successful')
            else:
                continue