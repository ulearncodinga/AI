import cv2
import numpy as np

#1.图像输入
image_np = cv2.imread('shudu.png')
#构2.建卷积核
# kernel = np.array([[-1,0,1],
#                    [-2,0,2],
#                    [-1,0,1]])#显示数值线
kernel = np.array([[-1,-2,-1],
                   [0,0,0],
                   [1,2,1]])#显示水平线

image_filter2D = cv2.filter2D(image_np,-1,kernel)
#3.图像显示
cv2.imshow('image_np',image_np)
cv2.imshow('image_filter2D',image_filter2D)
cv2.waitKey(0)