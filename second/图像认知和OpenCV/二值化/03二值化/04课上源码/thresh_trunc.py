# 截断阈值法


import cv2
import numpy as np


# 首先还是使用opencv去读取一张图片
image_np = cv2.imread('./flower.png')


# 2. 将读取到的彩色图进行灰度化
image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# 定义阈值法所需要的阈值
thresh = 150

# 定义阈值法所需要的最大值
maxval = 255

# 直接使用opencv的threshold函数去二值化
ret, image_thresh = cv2.threshold(image_gray, thresh, maxval, cv2.THRESH_TRUNC)

# # 为了能够遍历到灰度图的所有像素点，所以需要获取灰度图的形状
# image_shape = image_gray.shape
#
# # 创建一个与灰度图大小相同的单通道图像，用来接收灰度图与阈值比较的结果
# image_thresh = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
#
# # 通过循环去遍历灰度图中的所有的像素点并和阈值进行比较
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         # 让灰度图中的像素点的像素值与阈值进行比较
#         # 如果像素值超过阈值就设置为阈值
#         if image_gray[i, j] > thresh:
#             image_thresh[i, j] = thresh
#         # 否则的话，就不变
#         else:
#             image_thresh[i, j] = image_gray[i, j]

# 显示图像
# 显示灰度图与截断阈值法处理过的二值化图
cv2.imshow('image_gray', image_gray)
cv2.imshow('image_thresh', image_thresh)
cv2.waitKey(0)