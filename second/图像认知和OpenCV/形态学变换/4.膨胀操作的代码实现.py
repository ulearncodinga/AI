import cv2

#1.读取图像
image_np = cv2.imread('./test.png')

#2.灰度化
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

#3.二值化
ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)

#4.构建核
kernel = cv2.getStructuringElement(cv2.MORPH_RECT,(5,5))

#5.膨胀操作

#必须准备的两个参数
# 1.要膨胀的二值化图像
# 2.构建好的结构化元素或者说是核
image_dilate = cv2.dilate(image_thresh,kernel)

#6.显示图像
cv2.imshow('image_thresh',image_thresh)
cv2.imshow('image_dilate',image_dilate)

cv2.waitKey(0)