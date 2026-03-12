''''''
'''''ROI：Region of Interest，翻译过来就是感兴趣的区域。什么意思呢？比如对于一个人的照片，假如我
们要检测眼睛，因为眼睛肯定在脸上，所以我们感兴趣的只有脸这部分，其他都不care，所以可以单独
把脸截取出来，这样就可以大大节省计算量，提高运行速度。
还记得Numpy这个库吗？我们在使用OpenCV进行读取图像时，图像数据会被存储为Numpy数组，这
也意味着我们可以使用Numpy数组的一些操作来对图像数据进行处理，比如切片。而本实验的原理也是
基于Numpy数组的切片操作来完成的，因此在对应的组件中就需要填我们要切割的ROI区域的坐标来完
成ROI切割操作。'''
# Numpy 3维数组:
# 第一维:高度
# 第二维:宽度
# 第三维:像素


'''
先操作行
再操作列

步骤:
图片输入 -> 图片切割 -> 图片输出
'''
import cv2

if __name__ == "__main__":
    path = "./lena.png"
    try:
        image_np = cv2.imread(path)
        (w, h, _) = image_np.shape
        x_min, x_max = 150, 270
        y_min, y_max = 150, 290
        if x_max - x_min > w or y_max - y_min > h:
            raise OverflowError("delta_x or delta_y is overflow!")
        img_rec = cv2.rectangle(image_np, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)
        # 提取目标中的感兴趣区域
        ROI_img = image_np[y_min: y_max, x_min: x_max]
        cv2.imshow("ROI_img", ROI_img)
        cv2.waitKey(0)
    except Exception as e:
        # 返回出错内容
        print({"return": "error", "error_result": str(e)})
