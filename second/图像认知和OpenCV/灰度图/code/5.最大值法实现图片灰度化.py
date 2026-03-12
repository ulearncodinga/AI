import cv2
import numpy as np

#1. 使用cv2.imread()去读取一张彩色图
image_np = cv2.imread('./flower.png')

#2.获取数组的形状
image_shape = image_np.shape
#3.创造一个跟彩色图宽和高相同的单通道图像
image_gray = np.zeros((image_shape[0],image_shape[1]),dtype=np.uint8)

#4.遍历彩色图像中所有的像素点,并去除像素点的三个通道中值最大的那个
for i in range(image_shape[0]):
    for j in range(image_shape[1]):
        image_gray[i][j] = max(image_np[i,j][0],image_np[i,j][1],image_np[i,j][2])

#5.显示图像
cv2.imshow('image_np',image_np)
cv2.imshow('image_gray',image_gray)

cv2.waitKey(0)