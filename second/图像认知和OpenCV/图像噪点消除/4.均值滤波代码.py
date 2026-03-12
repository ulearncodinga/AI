import cv2

#1.读取图像
image_np = cv2.imread('./lena.png')

#2.获取卷积核(滤波器)并进行均值滤波操作
# cv2.blur
"""(src,
 dst,
 ksize,
 锚点:anchor(默认值-1,-1),
 borderType)
"""
image_blur = cv2.blur(image_np,(3,3))


#3.显示结果
cv2.imshow('image_np',image_np)
cv2.imshow('image_blur',image_blur)
cv2.waitKey(0)