# 这个文件的作用是用来对图像中的某些区域进行切割的

# 导入opencv库，方便读取和显示图像
import cv2

# 1. 读取我们要切割的图像
image_np = cv2.imread('./lena.png')

# 获取图像的高度和宽度，方便后续进行判断
height, width = image_np.shape[0], image_np.shape[1]

try:
    # 2. 切割感兴趣的区域
    # 人为指定我们要切割的区域
    x_min, x_max = 270, 400
    y_min, y_max = 270, 400

    # 对要切割的区域的范围进行判断
    if not (x_min >= 0 and x_max <= width and y_min >= 0 and y_max <= height):
        raise OverflowError("x_min or x_max or y_min or y_max overflow!!")

    # 使用cv2.rectangle去画一个矩形框，方便我们去调整感兴趣的区域的范围
    cv2.rectangle(image_np, (x_min - 2, y_min - 2), (x_max + 2, y_max + 2), (0, 0, 255), 2)


    # 3. 使用np数组的切片操作对图像进行切割
    image_roi = image_np[y_min: y_max, x_min: x_max]
    # 4. 显示结果
    cv2.imshow('image_np', image_np)
    cv2.imshow('image_roi', image_roi)
    cv2.waitKey(0)
except Exception as e:
    print(e)
