import cv2
image_np = cv2.imread('./lena.png')
image_bilateral = cv2.bilateralFilter(image_np,5,75,75)

#bilateralFilter:
'''
src
d:过滤期间使用的每个像素领域的直径
sigmaColor
sigmaSpace
borderType
'''

cv2.imshow('image_np',image_np)
cv2.imshow('image_bil',image_bilateral)
cv2.waitKey(0)