''''''
'''
双边滤波会保留更多的边缘信息

双边滤波采用了两个高斯滤波的结合，一个负责计算空间邻近度的权值（也就是空域信息），
也就是上面的高斯滤波器，另一个负责计算像素值相似度的权值（也就是值域信息），
也是一个高斯滤波器

'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    no_noise_image = cv2.bilateralFilter(image_np, 9, 9, 9)  # 双边滤波
    # 返回处理正确后的内容
    cv2.imshow("image_np", image_np)
    cv2.imshow("no_noise_image", no_noise_image)
    cv2.waitKey(0)
