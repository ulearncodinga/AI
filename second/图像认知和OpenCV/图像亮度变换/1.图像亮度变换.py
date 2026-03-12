import cv2
import numpy as np
#1,读取
image_np = cv2.imread('./flower.png')

#2.对Numpy数组进行数值运算亮度变换
# 为了保证图像正常显示,数据处于0-255之内且数据类型为整形
# clip函数:是一个截断函数,你需要它提供要截取的范围,提供截取的对象
image_brightness = np.uint8(np.clip(image_np *1.0 + 50,0,255))
# image_brightness = image_np + 200


cv2.imshow('image_np',image_np)
cv2.imshow('image_brightness',image_brightness)
cv2.waitKey(0)