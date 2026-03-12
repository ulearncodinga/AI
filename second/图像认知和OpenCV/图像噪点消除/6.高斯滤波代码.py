import cv2

image_np = cv2.imread('./lena.png')

image_gaussianBlur = cv2.GaussianBlur(image_np,(3,3),1)
# src,ksize,sigmaX,dst,sigmaY,borderType


cv2.imshow('image_np',image_np)
cv2.imshow('image_gauss',image_gaussianBlur)
cv2.waitKey(0)