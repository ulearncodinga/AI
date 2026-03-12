# 使用反阈值法去二值化图像

import cv2
import numpy as np

# 1. 读取要二值化的彩色图
image_np = cv2.imread('./flower.png')

# 2. 将读取到的彩色图进行灰度化
image_gray = cv2.cvtColor(image_np, cv2.COLOR_BGR2GRAY)

# 获取灰度图的形状
# image_shape = image_gray.shape

# 创建一个二值化图模板
# image_thresh = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
# 3.遍历灰度图的所有像素点，并将其跟阈值进行比较

# 设置阈值
thresh = 127

# 设置最大值
maxval = 255

# 使用opencv的接口  cv2.threshold去二值化图像
ret, image_thresh = cv2.threshold(image_gray, thresh, maxval, cv2.THRESH_BINARY_INV)

# 使用嵌套的循环去遍历灰度图中的所有像素点，并将其跟阈值进行比较，
# 将比较的结果存放到二值化图中
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         # 如果灰度图的第i行第j列的像素点的像素值比thresh要大
#         # 就将其设置为0
#         if image_gray[i, j] > thresh:
#             image_thresh[i, j] = 0
#         # 否则的话，就设置为max
#         else:
#             image_thresh[i, j] = maxval


# 4. 显示结果
# cv2.imshow('image_gray', image_gray)
# cv2.imshow('image_thresh', image_thresh)
cv2.imshow('image_thresh', image_thresh)
cv2.waitKey(0)


