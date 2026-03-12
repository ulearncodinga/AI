

# 导入OpenCV的库
import cv2


# 1.图片输入，读取一张图片
image_np = cv2.imread('./lena.png')

# 2. 图片缩放
# dsize和fx、fy不能同时使用，如果同时出现，会以dsize的标准进行缩放
# 如果想要使用resize函数，就必须填入两个参数：src和dsize
# 如果不想使用dsize，赋为None就行。
image_resize = cv2.resize(image_np, dsize=None, fx=0.1, fy=1, interpolation=cv2.INTER_LINEAR)

# 3. 显示图像
cv2.imshow('image_np', image_np)
cv2.imshow('image_resize', image_resize)
cv2.waitKey(0)
