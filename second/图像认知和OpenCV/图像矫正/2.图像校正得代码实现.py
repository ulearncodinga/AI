#导入opencv库,方便调用函数
import cv2
import numpy as np
#1.读取图片
image_np = cv2.imread('./test.png')
#获取图像大小
img_shape = image_np.shape

#2.定义原始图像中四个顶点的坐标
points1 = np.float32([[210,115],[685,170],[135,400],[650,460]])
#定义目标图像中,这四个顶点坐标所对应得位置
points2 = np.float32([[0,0],[img_shape[1],0],[0,img_shape[0]],[img_shape[1],img_shape[0]]])



cv2.line(image_np, points1[0].astype(np.int64).tolist(),points1[1].astype(np.int64).tolist(),(0,0,255),1,lineType=cv2.LINE_AA)
cv2.line(image_np, points1[0].astype(np.int64).tolist(),points1[2].astype(np.int64).tolist(),(0,0,255),1,lineType=cv2.LINE_AA)
cv2.line(image_np, points1[3].astype(np.int64).tolist(),points1[1].astype(np.int64).tolist(),(0,0,255),1,lineType=cv2.LINE_AA)
cv2.line(image_np, points1[3].astype(np.int64).tolist(),points1[2].astype(np.int64).tolist(),(0,0,255),1,lineType=cv2.LINE_AA)





#3.获取透视变换矩阵
M = cv2.getPerspectiveTransform(points1,points2)#得到透视变换矩阵

#4.进行透视变换
image_warp = cv2.warpPerspective(image_np,M,(img_shape[1],img_shape[0]))

#5.显示
cv2.imshow('image_np',image_np)
cv2.imshow('image_warp',image_warp)
cv2.waitKey(0)