import cv2
import numpy as np

image_np = cv2.imread('./picture.png')
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
image_shape = image_np.shape
#Canny边缘检测
image_canny = cv2.Canny(image_gray,30,70)

#霍夫圆检测
# circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT,1,20,param1=70,param2=50)
# circles = np.int0(np.around(circles))
circles = cv2.HoughCircles(image_gray,cv2.HOUGH_GRADIENT_ALT,1,20,param1=70,param2=0.9)
circles = np.int0(np.around(circles))

#创建一个模板
image_Circle = np.zeros(image_shape,dtype = np.uint8)

for circle in circles:
    x,y,radius = circle[0]
    cv2.circle(image_Circle,(x,y),radius,(0,0,255),2)

cv2.imshow('image_np',image_np)
cv2.imshow('image_Circle',image_Circle)
cv2.waitKey(0)