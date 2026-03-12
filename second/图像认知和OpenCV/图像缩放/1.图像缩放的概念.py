# 图片输入 -> 图片缩放 -> 插值方法 -> 图片输出
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    image_np = cv2.imread(path)
    resize_image = cv2.resize(image_np, None, fx=0.6, fy=0.6, interpolation=cv2.INTER_LINEAR)  # 对图片进行缩放
    cv2.imshow('image_np',image_np)
    cv2.imshow("resize_image", resize_image)
    cv2.waitKey(0)
# 核心函数:resize
