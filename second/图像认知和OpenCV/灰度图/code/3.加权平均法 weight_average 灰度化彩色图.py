#如何去使用opencv去读取一张图片
# opencv 用 BGR 的顺序存储
import cv2
import numpy as np

#使用opencv去读取一张图片,在opencv中使用cv2.imread去读取一张图片
#cv2.imread():两个参数,第一个是要读取的图片的位置及名称(名称要包括文件的后缀名)
#第二个参数是指定读取进来的图片的格式,默认使用BGR彩色图的格式,如果有特殊需要可以去opencv官网查看

image_np = cv2.imread('flower.png')#返回一个数组
# print(type(image_np))

#使用opencv的接口去灰度化一张图像
cv_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)




#shape:是Ndarray的一个属性,用来查看数组的形状
#shape读取到的形状与图像实际宽和高是相反的,shape[0]代表的是图像的高度
#shape[1]代表的是图像的宽度
image_shape = image_np.shape

#创建一个单通道的全0数组,此时需要创建一个与原图大小相同的单通道数组
#zeros:按照高和宽的顺序来创建的
#image_gray就是我们创建的一个灰度图模板,像素值全是0
image_gray = np.zeros((image_shape[0],image_shape[1]),dtype = np.uint8)

# 定义权重
# weight_red = 0.299
# weight_green = 0.587
# weight_blue = 0.114
#把每一个点的三通道值乘以权重在加和
#遍历彩色图像,对彩色图像中每个像素点都进行加权平均的操作
#求出每个像素点的灰度值,然后将得到的灰度值赋值给image_gray

#通过嵌套循环,让我们能遍历到图片中的所有像素点
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         #遍历到所有像素点之后开始进行加权平均的计算
#         image_gray[i][j] = round(image_np[i,j][0] * weight_blue + image_np[i,j][1] * weight_green + image_np[i,j][2] * weight_red)

#显示彩色图
cv2.imshow('image_np',image_np)
#使用cv2.imshow()去显示一下image_gray
cv2.imshow('cv_gary',cv_gray)
#使用cv2.waitKey(0)将图像固定下来
cv2.waitKey(0)

