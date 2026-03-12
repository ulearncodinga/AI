import cv2
import numpy as np
image_np = cv2.imread('./picture.png')

image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
ret,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_OTSU +cv2.THRESH_BINARY_INV)
image_contour = image_np.copy()

contours,hierarchy = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image_contour,contours,-1,(0,0,255),2)


#绘制最小外接圆
# cv2.minEnclosingCircle (points,center,radius)
#通过cv2.EnclosingCircle 函数获取最小外接圆的圆心和半径
for cnt in contours:
    (x,y),radius = cv2.minEnclosingCircle(cnt)
    #返回的圆心坐标不是整数,需要取整
    (x,y,radius) = np.int0((x,y,radius))

    #画圆 cv2.circle()
    cv2.circle(image_contour,(x,y),radius,(255,0,0),2)
cv2.imshow('image_np',image_np)
cv2.imshow('image_contours',image_contour)
cv2.waitKey(0)