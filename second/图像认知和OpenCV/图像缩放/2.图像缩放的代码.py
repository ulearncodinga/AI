# resize的使用
import cv2
#1.读取一张图片
image_np = cv2.imread('./lena.png')
#2.图片缩放resize
''''''
'''
cv2.resize(src,dsize[,dst[,fx[,fy[,]]]])
fx,fy与dsize不同时生效,如果同时出现,会以dsize的标准进行缩放.且fx,fy 和 dsize不同时为0
fx,fy适合比例缩放
如果想要使用resize函数,就必须填入两个参数:src和dsize
如果不想使用dsize,赋值为None就行
'''
# image_resize = cv2.resize(image_np,(100,200),interpolation=cv2.INTER_LINEAR)
image_resize = cv2.resize(image_np,dsize=None,fx=0.5,fy=1.5,interpolation=cv2.INTER_LINEAR)
#3.显示图像
cv2.imshow('image_np',image_np)
cv2.imshow('image_resize',image_resize)
cv2.waitKey(0)