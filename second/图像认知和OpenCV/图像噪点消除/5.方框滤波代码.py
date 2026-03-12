import cv2

image_np = cv2.imread('./lena.png')

#直接进行方框滤波
# boxFilter
image_filter = cv2.boxFilter(image_np,-1,(3,3))
# image_filter = cv2.boxFilter(image_np,-1,(3,3),normalize=False)

cv2.imshow('image_np',image_np)
cv2.imshow('image_filter',image_filter)
cv2.waitKey(0)