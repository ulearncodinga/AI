

# 导入OpenCV库，方便调用函数
import cv2
import numpy as np

# 1. 读取图片
image_np = cv2.imread('./test.png')

# 获取图像的大小
img_shape = image_np.shape


# 2. 定义原始图像中的四个顶点的坐标
points1 = np.array([[200, 100], [700, 150], [140, 400], [650, 460]])
# 定义目标图像中 这四个顶点坐标所对应的位置
points2 = np.array([[0, 0], [img_shape[1], 0], [0, img_shape[0]], [img_shape[1], img_shape[0]]])

cv2.line(image_np, points1[0].astype(np.int64).tolist(), points1[1].astype(np.int64).tolist(), (0, 0, 255), 1, lineType=cv2.LINE_8)
cv2.line(image_np, points1[0].astype(np.int64).tolist(), points1[2].astype(np.int64).tolist(), (0, 0, 255), 1, lineType=cv2.LINE_8)
cv2.line(image_np, points1[3].astype(np.int64).tolist(), points1[1].astype(np.int64).tolist(), (0, 0, 255), 1, lineType=cv2.LINE_8)
cv2.line(image_np, points1[3].astype(np.int64).tolist(), points1[2].astype(np.int64).tolist(), (0, 0, 255), 1, lineType=cv2.LINE_8)




# 3. 获取透视变换矩阵
M = cv2.getPerspectiveTransform(points1, points2)

# 4. 进行透视变换
image_warpPerspective = cv2.warpPerspective(image_np, M, (img_shape[1], img_shape[0]), flags=cv2.INTER_CUBIC)

# 5. 显示图像
cv2.imshow('image_np', image_np)
cv2.imshow('image_warpPerspective', image_warpPerspective)
cv2.waitKey(0)

