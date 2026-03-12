#对图像进行角度旋转(逆时针为正方向)
#在opencv中以逆时针反向作为正方向

#导入opencv库
import cv2

#1.读取一张图片
image_np = cv2.imread('./lena.png')
#定义旋转的角度和图片缩放的比例
angle = 45
scale = 0.5

#获取原始图像大小
image_shape = image_np.shape


# cv2.getRotationMatrix2D(center,angle,scale)
#构建旋转矩阵
M = cv2.getRotationMatrix2D((0,0),angle,scale)

#有了旋转矩阵之后就能进行图像旋转了
# cv2.warpAffine(src,M,dsize[,dst[,flags[,borderMode[,borderValue[]]]]])
    #第一个参数:要旋转的图像
    #第二个参数:通过cv2.getRotationMatrix2D 获取到的旋转矩阵
    #第三个参数:输出图像的大小

# cv2.invertAffineTransform()#反转旋转
image_rotation = cv2.warpAffine(image_np,M,(image_shape[1],image_shape[0]),flags=cv2.INTER_LINEAR,borderMode=cv2.BORDER_CONSTANT,borderValue=(0,0,0))

#图像显示
cv2.imshow("image_np",image_np)
cv2.imshow("image_rotation",image_rotation)
cv2.waitKey(0)