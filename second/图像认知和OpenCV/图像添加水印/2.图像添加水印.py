#导入opencv库
import cv2

#1.读取原始图像,和模板图
image_np = cv2.imread('./lena.png')
logo = cv2.imread('./photo.png')
cv2.imshow('image_np',image_np)
#获取logo图的大小
rows,cols = logo.shape[:2]
#截取原始图像的ROI区域
roi = image_np[:rows,:cols]
#2.对logo进行灰度化和二值化,制作掩膜
logo_gray = cv2.cvtColor(logo,cv2.COLOR_BGR2GRAY)
ret,mask = cv2.threshold(logo_gray,127,255,cv2.THRESH_BINARY_INV)

#3.进行与运算
image_and = cv2.bitwise_and(roi,roi,mask=mask)

#4.图像融合
# cv2.add(src1,src2,dst,mask,dtype)
dst = cv2.add(image_and,logo)
#5.将融合后的图像重新赋值到原始图像中
image_np[:rows,:cols] = dst

#6.显示结果
cv2.imshow('logo',logo)
cv2.imshow('image_and',image_and)
cv2.imshow('image_np1',image_np)
cv2.waitKey(0)

