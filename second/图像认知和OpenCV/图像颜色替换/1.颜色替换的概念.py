''''''
'''RGB->HSV->制造掩膜(mask) 

图片颜色替换
前一个实验中，我们已经能够识别到图像中的某一种颜色，那么我们就可以对识别到的颜色进行一个操
作，比如将其替换成别的颜色，其原理就是在得到原图的掩膜之后，对掩膜中的白色区域所对应的原图
中的区域进行一个像素值的修改即可。但是这其中还需要一些别的操作，接下来一一介绍。
1. HSV空间转换
首先还是先要将图片从RGB颜色空间转化为HSV颜色空间，方便后续的操作。HSV颜色空间指的是HSV
颜色模型，这是一种与RGB颜色模型并列的颜色空间表示法。RGB颜色模型使用红、绿、蓝三原色的强
度来表示颜色，是一种加色法模型，即颜色的混合是添加三原色的强度。而HSV颜色空间使用色调
（Hue）、饱和度（Saturation）和亮度（Value）三个参数来表示颜色，色调H表示颜色的种类，如红
色、绿色、蓝色等；饱和度表示颜色的纯度或强度，如红色越纯，饱和度就越高；亮度表示颜色的明暗
程度，如黑色比白色亮度低。


在“形态学变换”这一章节中我们了解了腐蚀与膨胀的工作原理，而开运算就是对图像先进行腐蚀操作，
然后进行膨胀操作。开运算可以去除二值化图中的小的噪点，并分离相连的物体
'''

import cv2
import numpy as np

if __name__ == "__main__":
    path = "./color.png"
    image_np = cv2.imread(path)
    hsv_image_np = cv2.cvtColor(image_np, cv2.COLOR_BGR2HSV)  # 转为HSV空间
    color_low = np.array([[0, 43, 46], [10, 255, 255]][0])
    color_high = np.array([[0, 43, 46], [10, 255, 255]][1])
    mask_image_np = cv2.inRange(hsv_image_np, color_low, color_high)  # 创建掩膜
    # 开操作
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    open_image_np = cv2.morphologyEx(mask_image_np, cv2.MORPH_OPEN, kernel)
    # 替换颜色
    image_np[open_image_np == 255] = (255, 0, 0)
    cv2.imshow("mask_image_np", mask_image_np)
    cv2.imshow("image_np", image_np)
    cv2.waitKey(0)
