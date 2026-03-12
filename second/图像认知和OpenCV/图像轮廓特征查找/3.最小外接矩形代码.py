import cv2
import numpy as np
image_np = cv2.imread('./picture.png')
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
ret,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_OTSU + cv2.THRESH_BINARY_INV)
image_contour = image_np.copy()

#寻找轮廓
contours,hierarty = cv2.findContours(image_binary,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

cv2.drawContours(image_contour,contours,-1,(0,0,255),2)

# 寻找最小外接矩形
# cv2.minAreaRect
#注意:cv2.minAreaRect:只能一个一个轮廓去返回结果

for cnt in contours:
    #cv2.minAreaRect 返回的是 角度,矩形中心,矩形的宽和高
    rect = cv2.minAreaRect(cnt)
    #由于rect 所包含的信息无法帮我们直接去绘制旋转矩形
    #所以需要使用cv2.boxPoints来帮我们获取旋转矩形的四个顶点坐标
    box = np.int0(cv2.boxPoints(rect))


    #根据四个顶点坐标将最小外接矩形所描绘出来
    #cv2.drawContours 的contours参数必须是contour级别的
    #此时的box是cnt级别的
    cv2.drawContours(image_contour,[box],-1,(255,0,0))

#7.结果显示

cv2.imshow('image_np',image_np)
cv2.imshow('image_contour',image_contour)
cv2.waitKey(0)
