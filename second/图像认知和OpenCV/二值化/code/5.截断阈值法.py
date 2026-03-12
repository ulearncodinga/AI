import cv2
import numpy as np

image_np = cv2.imread('./flower.png')
image_np = cv2.resize(image_np, (400, 400))

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

thresh = 150
maxval = 255


image_shape = image_gray.shape

image_thresh = np.zeros((image_shape[0],image_shape[1]),dtype = np.uint8)

'''
使用opencv的threshold进行二值化
'''
ret,image_thresh = cv2.threshold(image_gray,thresh,maxval,cv2.THRESH_TRUNC)

cv2.imshow('image_thresh',image_thresh)
cv2.waitKey(0)


























# #通过循环遍历灰度图中所有像素点
#
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         #让灰度图中的像素点与阈值进行比较
#         #如果像素值超过阈值就设置为阈值
#         if image_gray[i,j] > thresh:
#             image_thresh[i,j] = thresh
#             #否则不变
#         else:
#             image_thresh[i,j] = image_gray[i,j]
#
# #显示图像
# #显示灰度图与截断阈值法处理过的二值化图
# cv2.imshow('image_gray',image_gray)
# cv2.imshow('image_thresh',image_thresh)
#
# cv2.waitKey(0)