#对图像进行反转操作
#导入opencv库
import cv2
#1.读取图片
image_np = cv2.imread('./lena.png')

#2.镜像旋转
# cv2.flip(src,flipCode[,dst])

#3.使用filp函数去对图像进行镜像的翻转
'''第一个参数:要翻转的原始图像
第二个参数:标志位,
0:表示绕x轴进行上下翻转;
>0表示绕y轴进行左右翻转
<0表示绕x轴y轴各进行一次翻转
'''
image_flip = cv2.flip(image_np,0)

#4.显示图像
cv2.imshow('image_np',image_np)
cv2.imshow('image_flip',image_flip)
cv2.waitKey(0)