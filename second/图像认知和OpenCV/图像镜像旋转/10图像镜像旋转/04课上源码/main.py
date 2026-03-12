# 对图像进行翻转操作
import cv2

# 导入OpenCV的库，方便我们调用函数

# 1. 读取图片
image_np = cv2.imread('./lena.png')


# 2. 使用flip函数去对图像进行镜像的翻转
# cv2.flip： 对图像进行镜像翻转
# 第一个参数：要翻转的原始图像
# 第二个参数：标志位， 0：表示绕x轴进行上下翻转， >0:表示绕y轴进行左右翻转  <0:表示绕x轴和y轴各进行一次翻转
image_flip = cv2.flip(image_np, 0)

# 3. 输出，显示
cv2.imshow('image_np', image_np)
cv2.imshow('image_flip', image_flip)
cv2.waitKey(0)