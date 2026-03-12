import cv2
import numpy as np
image_np = cv2.imread('./picture.png')
image_shape= image_np.shape
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
#Canny边缘化
image_canny = cv2.Canny(image_gray,30,70)
lines= cv2.HoughLines(image_canny,0.8,np.pi/180,90)

image_HoughLines = np.zeros(image_shape,dtype = np.uint8)

for line in lines:
    rho,theta = line[0]
    cos_theta = np.cos(theta)
    sin_theta = np.sin(theta)
    #rho = x * cos_theta + y * sin_theta
    #y = (rho - x * cos_theta) / sin_theta
    x1,x2 = 0 , image_shape[1]
    y1 = int((rho - x1 * cos_theta) / sin_theta)
    y2 = int((rho - x2 * cos_theta) / sin_theta)

    cv2.line(image_HoughLines,(x1,y1),(x2,y2),(0,0,255))

cv2.imshow('image_np',image_np)
cv2.imshow('image_HoughLines',image_HoughLines)
cv2.waitKey(0)