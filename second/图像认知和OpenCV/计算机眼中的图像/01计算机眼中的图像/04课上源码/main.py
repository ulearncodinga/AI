
# 使用numpy库来创建三维数组
import numpy as np

# 使用OpenCV库来进行查看图像及其他操作
import cv2

# 使用matplotlib库来一次性展示多个图像
import matplotlib.pyplot as plt

# 创建一个700 * 700 的三维数组（彩色图）
# 使用np.zeros创建一个数值全为0的数组，对应一张黑色图
# uint8: unsigned int 无符号整数 0-255
image = np.zeros((700, 700, 3), dtype=np.uint8)

# 用来代表分割出来的矩形的大小
block_size = 100

# 使用一个嵌套循环对黑色图像进行分割
# 使用i来代表每一行
for i in range(0, 700, block_size):
    # 使用j来代表每一列
    for j in range(0, 700, block_size):
        # i的取值：0，100，200，300，400，500，600
        # j的取值：0，100，200，300，400，500，600
        # 使用三维数组切片修改的方式去修改像素值
        # 对每一行来说，需要选中的是第i行里的所有像素
        # image[i, :, :] = (255, 255, 255)
        # image[:, j, :] = (255, 255, 255)
        #
        # # 根据观察图像所得，i和j的取值都不能为0和600
        # if i != 0 and j != 0 and i != 600 and j != 600 and (i == j or i + j == 600):
        #     # 在OpenCV中，颜色的顺序是BGR
        #     image[i:i+block_size, j:j+block_size, :] = (0, 0, 255)

        # OpenCV中，以右为x轴正方向，以下为y轴正方向
        # 对应到我们的程序中，就是以j为x轴坐标，以i为y轴坐标
        top_left = (j, i)
        bottom_right = (j + block_size - 1, i + block_size - 1)
        # 如果满足条件，就使用cv2.rectangle()来画红色的实心矩形
        if i != 0 and i != 600 and (i == j or i + j == 600):
            cv2.rectangle(image, top_left, bottom_right, (0, 0, 255), -1)
        # 如果不满足条件，就需要画白线
        else:
            cv2.rectangle(image, top_left, bottom_right, (255, 255, 255), 2)


# matplotlib的颜色顺序是RGB，所以需要进行BGR到RGB的转换
# 在image_rgb里存放的是RGB顺序的图像，在image里存放的是BGR顺序的图像
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# 使用plt.subplot()让本图处于第1行第2列的位置上，让其他图处于第二行的位置
plt.subplot(232)
# imshow是用来展示图像的
plt.imshow(image_rgb)
# title是用来给图像添加标签
plt.title('Original Image')
# 不显示坐标轴
plt.axis("off")
# show函数：用来显示图像
# plt.show()


# 获取原图中的三个通道的像素值
# 有两种方法
# 第一种使用数组切片的方式去获取
b = image[:, :, 0]
g = image[:, :, 1]
r = image[:, :, 2]

# 第二种：使用cv2.split()函数去分割
# b, g, r = cv2.split(image)

# 创建新的图像，用来展示三通道图
blue_channel = np.zeros((700, 700, 3), dtype=np.uint8)
green_channel = np.zeros((700, 700, 3), dtype=np.uint8)
red_channel = np.zeros((700, 700, 3), dtype=np.uint8)


# 将获取到的原图像的三通道的像素值，覆盖到新创建的三个图像的对应的通道
blue_channel[:, :, 0] = b
green_channel[:, :, 1] = g
red_channel[:, :, 2] = r

# 将BGR转换为RGB
blue_channel_rgb = cv2.cvtColor(blue_channel, cv2.COLOR_BGR2RGB)
green_channel_rgb = cv2.cvtColor(green_channel, cv2.COLOR_BGR2RGB)
red_channel_rgb = cv2.cvtColor(red_channel, cv2.COLOR_BGR2RGB)

# plt.subplot() 用来对要展示的图像进行布局
# 131:创建1行3列的布局，并且本图像处于第1个位置
plt.subplot(234)
# 用来显示图像，但不能单独使用，需要配合plt.show()来显示
plt.imshow(blue_channel_rgb)
# 设置标签
plt.title("Blue Channel")
# 设置不显示坐标轴
plt.axis('off')

# 132:还是1行3列的布局，本图处于第2个位置
plt.subplot(235)
# 用来显示图像，但不能单独使用，需要配合plt.show()来显示
plt.imshow(green_channel_rgb)
# 设置标签
plt.title("Green Channel")
# 设置不显示坐标轴
plt.axis('off')

# 133:还是1行3列的布局，本图处于第3个位置
plt.subplot(236)
# 用来显示图像，但不能单独使用，需要配合plt.show()来显示
plt.imshow(red_channel_rgb)
# 设置标签
plt.title("Red Channel")
# 设置不显示坐标轴
plt.axis('off')

# plt.tight_layout()：作用就是合理布局图像
plt.tight_layout()

plt.show()

# # 注意：窗口的名字尽量不要重名，否则后面的图像会把前面的图像给覆盖掉
# cv2.imshow('image', image)
# # 当参数为0时，可以展示我们的图像
# cv2.waitKey(0)

