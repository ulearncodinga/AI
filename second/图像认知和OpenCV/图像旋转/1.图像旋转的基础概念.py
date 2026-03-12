''''''
'''
1.旋转矩阵
    1.1 旋转
    1.2 平移
2.插值方法
    2.1最近邻插值 k=1最近邻的knn算法(会有严重马赛克现象,效果不好,计算效率最高,效果最差)
    2.2双线性插值 
    2.3其他插值(双三次插值,蓝索思插值)
    
3.边界填充
    3.1常数填充
    3.2边界复制
    
图片输入 -> 单点旋转 -> 图片旋转 -> 插值方法 -> 边缘填充方法 -> 图片输出
'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    img_shape = image_np.shape
    # 对图片进行旋转
    M = cv2.getRotationMatrix2D((img_shape[0] / 2, img_shape[1] / 2), 45, 0.5)
    rotation_image = cv2.warpAffine(image_np, M, (img_shape[0], img_shape[1]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)
    cv2.imshow("rotation_image", rotation_image)
    cv2.waitKey(0)
