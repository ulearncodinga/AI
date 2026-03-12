# 图像镜像旋转,也称为图像翻转
# 输入->镜像反转 -> 输出


''' flipcode < 0'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    # 对图片进行镜像操作
    mirroring_image = cv2.flip(image_np, -1)
    cv2.imshow("mirroring_image", mirroring_image)
    cv2.waitKey(0)
'''flipcode = 0'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    # 对图片进行镜像操作
    mirroring_image = cv2.flip(image_np, 0)
    cv2.imshow("mirroring_image", mirroring_image)
    cv2.waitKey(0)


''' flipcode > 0'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    # 对图片进行镜像操作
    mirroring_image = cv2.flip(image_np, 1)
    cv2.imshow("mirroring_image", mirroring_image)
    cv2.waitKey(0)
