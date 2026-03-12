#对二值化图像进行腐蚀操作

import cv2
#1.读取一张要操作的彩色图像
image_np = cv2.imread('./test.png')
#2.灰度化
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
#3.将灰度图转化为腐蚀操作所需要的二值化图'
ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)
#4.进行腐蚀操作:
#4.1构建腐蚀操作所需要用到的核
# cv2.getStructuringElement:用来生成形态变换所需要用到的核
#第一个参数:指定核的形状:cv2.MORPH_CROSS表示十字形,cv2.MORPH.RECT 表示矩形  cv2.MORPH_ELLTPS 表示椭圆形
#第二个参数:指定核的大小  :以元组的形式传递2
#第三个参数:指定十字形核的核值分布
kernel = cv2.getStructuringElement(cv2.MORPH_CROSS,(5,5))
# print(kernel)


# #4.2去进行滑动计算
image_erode = cv2.erode(image_thresh,kernel)#必须准备的两个参数:要操作的二值化图,核

#5.显示图像
cv2.imshow('image_thresh',image_thresh)
cv2.imshow('image_erode',image_erode)

cv2.waitKey(0)