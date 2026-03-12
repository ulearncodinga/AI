# 对图像进行任意角度的旋转
# 在OpenCV中，以逆时针方向作为正方向


# 导入OpenCV库
import cv2


# 1. 读取一张图片
image_np = cv2.imread('./lena.png')

# 定义旋转的角度和图片缩放的比例
angle = 45
scale = 0.5

# 获取原始图像的大小
image_shape = image_np.shape


# 构建旋转矩阵
# cv2.getRotationMatrix2D: 构建一个2*3的旋转矩阵，包含了图像的旋转、平移与缩放
# 第一个参数：旋转中心
# 第二个参数：旋转角度
# 第三个参数：缩放比例
M = cv2.getRotationMatrix2D((image_shape[1] / 2, image_shape[0] / 2), angle, scale)

# 有了旋转矩阵之后，就能直接去进行图像旋转了
# cv2.warpAffine：对图像进行旋转
# 第一个参数：要旋转的图像
# 第二个参数：通过cv2.getRotationMatrix2D 获取到的旋转矩阵
# 第三个参数：输出图像的大小
image_rotation = cv2.warpAffine(image_np, M, (image_shape[1], image_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_WRAP)

# 显示旋转的结果
cv2.imshow('image_np', image_np)
cv2.imshow('image_rotation', image_rotation)
cv2.waitKey(0)