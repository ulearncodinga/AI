# 使用OTSU去计算合适的阈值,并且结合阈值法进行二值化
import cv2
import numpy as np

#1.读取二值化的彩色图
image_np = cv2.imread('./flower.png')

#2.进行灰度化
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

thresh = 127
maxval = 255




#注意OTSU在使用的时候,需要配合其他的二值化方法去进行,其模式就是cv2.THRESH_OTSU + 要二值化的方法的参数
ret,image_thresh = cv2.threshold(image_gray,thresh,maxval,cv2.THRESH_BINARY+cv2.THRESH_OTSU)

#使用cv2.imwrite()去保存图像
#第一个参数是保存的图像名称和路径,第二个参数是要保存的图像
cv2.imwrite('thresh.png',image_thresh)

#显示图像
cv2.imshow('image_thresh',image_thresh)
cv2.waitKey(0)






#
#
# ''''''
# import numpy as np
#
# '''
# 手写OTSU
# '''
# import cv2
#
# image_np = cv2.imread('./flower.png')
# #使用cv2.resize调整图片大小
# image_np = cv2.resize(image_np,(100,100))
#
# image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)
# image_shape = image_gray.shape
#
# #设置最大值
# maxval = 255
#
# #生成一个二值化模板图
# image_thresh = np.zeros((image_shape[0],image_shape[1]),dtype = np.uint8)
#
#
#
# #使用np数组的min()函数去获取数组中的最小值
# min_value = image_gray.min()
# #使用max()去获取数组中的最大值
# max_value = image_gray.max()
#
# #定义计算最大类间方差公式
# n_0 = 0
# n_1 = 0
# w_0 = 0
# w_1 = 0
# u_0 = 0
# u_1 = 0
# u = 0
# rows = image_shape[0]
# cols = image_shape[1]
#
# forepix = np.int64(0)
# backpix = np.int64(0)
# pix = np.int64(0)
#
# #定义一个字典,用来存储每一个阈值所对应的最大类间方差,方便后面获取合适的阈值
# var = {}
#
# #控制阈值t取值的循环,取值范围是灰度图中最小像数值加1到最大像素值减1
# for t in range(min_value +1,max_value,1):
#     # 定义一个列表用来存储前景像素点
#     foreground = []
#
#     # 定义一个列表用来存储前景像素点
#     background = []
#
#
#     #定义一个变量用来存储前景的像素值的总数
#     # forepix = 0
#
#     #定义一个变量来用来存储背景的像素值的总数
#     # backpix = 0
#
#     #定义一个变量用来求灰度图中所有像素值的和
#     # pix = 0
#
#
#     #区分前景和背景
#     # 使用嵌套的for循环去遍历灰度图,区分当前阈值下,哪些是前景点哪些是背景点
#     for i in range(image_shape[0]):
#         for j in range(image_shape[1]):
#             #将灰度图的每个像素点和阈值进行比较,如果大于阈值就是前景像素点
#             if image_gray[i,j] > t:
#                 foreground.append([i,j])
#                 #求前景像素点的总像素值
#                 forepix += image_gray[i,j]
#
#                 #将该像素点加到pix里,用来统计总像素值
#                 pix += image_gray[i,j]
#             else:
#                 background.append([i,j])
#                 #求背景像素点的总像素值
#                 backpix += image_gray[i,j]
#                 # 将该像素点加到pix里,用来统计总像素值
#                 pix += image_gray[i, j]
#
#     #获取前景像素点数
#     n_0 = len(foreground)
#     #获取背景像素点数
#     n_1 = len(background)
#
#     #通过计算获取w0
#     w_0 = n_0 / (image_shape[0] * image_shape[1])
#     #通过计算w1
#     w_1 = n_1 / (image_shape[0] * image_shape[1])
#
#     #通过计算获取前景的总像素值
#     u_0 = float(forepix / n_0)
#
#     #通过计算获取背景的总像素值
#     u_1 = float(backpix / n_1)
#
#
#     #通过计算获取整幅图的平均像素值
#     u = float(pix / (image_shape[0] * image_shape[1]))
#
#     #通过最大类间方差公式去计算当前阈值下的最大的类间方差结果
#     g = w_0 *((u_0- u) ** 2) +w_1* ((u_1- u) ** 2)
#
#     #将获取到的最大类间方差值和对应的阈值一块存储到字典中,方便后续选出最大值
#     var[t] = g
#
#
#
# #for循环结束后就可以去比较最大类间方差
# #寻找字典中最大的值所对应的键
# thresh = max(var,key=var.get)
#
# #使用嵌套循环进行二值化操作
# for i in range(image_shape[0]):
#     for j in range(image_shape[1]):
#         if image_gray[i,j] > thresh:
#             image_thresh[i,j] = maxval
#         else:
#             image_thresh[i,j] =0
#
#
# cv2.imshow('image_thresh',image_thresh)
# cv2.waitKey(0)
#

