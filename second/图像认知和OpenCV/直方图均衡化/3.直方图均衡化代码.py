import cv2
import numpy as np
#做一个函数用来统计和绘制直方图
def calcAndDrawHist(image_gray):
    #注意的是:所有参数都需要转换为列表的形式去传递
    hist = cv2.calcHist([image_gray],[0],None,[256],[0,256])

    #为了下面可以进行归一化操作,需要找到hist里面最大值
    minVal,maxVal,minLoc,maxLoc = cv2.minMaxLoc(hist)



    #创建一个256*256大小的模板图
    histImg = np.zeros((256,256,3),dtype=np.uint8)

    for h in range(256):
        #在循环里对每一列数据做一次归一化处理,防止超出模板图的范围
        intensity = int(256 * hist[h] / maxVal)
        cv2.line(histImg,(h,256),(h,256-intensity),(255,0,0))

    return histImg

#1.读取图像
image_np = cv2.imread('./picture.png')

#2.灰度化
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

#3.绘制直方图
# calcHist:计算一个或多个数组的直方图
# image,channels,mask,histSize,ranges,hist,accumulate
hist_image = calcAndDrawHist(image_gray)

#4.1直方图均衡化
# image_equalizeHist =cv2.equalizeHist(image_gray)
# image_equalizeHist_image = calcAndDrawHist(image_equalizeHist)

#4.2对比度受限制的自适应直方图均衡化的使用
clahe = cv2.createCLAHE(2,(8,8))
image_clahe = clahe.apply(image_gray)
image_clahe1 = calcAndDrawHist(image_clahe)



cv2.imshow('image_np',image_np)
cv2.imshow('hist_image',hist_image)
cv2.imshow('image_clahe1',image_clahe1)
# cv2.imshow('image_equalizeHist',image_equalizeHist)
# cv2.imshow('image_equalizeHist_image',image_equalizeHist_image)

cv2.waitKey(0)