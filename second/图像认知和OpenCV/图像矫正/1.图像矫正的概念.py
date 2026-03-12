''''''
'''
图像矫正实际上是使用透视变换的原理去矫正图像中的某些物体的被观察视角

透视变换：也成为单应性变换，是一种二维平面间的投影变换，能够将一幅图像
中的点映射到另一幅图像中相对应的点。它可以看作是一种特殊的仿射变换，不
仅保持了直线的平行性和共线性，还能够处理图像中的透视效果，即进大远小。
这意味着透视变换能够模拟相机视角的变化，使得从不同角度拍摄的图像能够对
齐。
乘的是3*3的透视变换矩阵

图片输入->左边选取->获取透视变换矩阵->透视变换->插值方法->边缘填充方法->图片输出

'''
import cv2
import numpy as np

if __name__ == "__main__":
    path = "./test.png"
    image_np = cv2.imread(path)
    img_shape = image_np.shape
    # 原图中卡片的四个角点
    pts1 = np.float32([[210, 115], [685, 170], [135, 400], [650, 460]])

    img_line = image_np
    cv2.line(img_line, pts1[0].astype(np.int64).tolist(), pts1[1].astype(np.int64).tolist(), (0, 0, 255), 2,
             cv2.LINE_AA)
    cv2.line(img_line, pts1[0].astype(np.int64).tolist(), pts1[2].astype(np.int64).tolist(), (0, 0, 255), 2,
             cv2.LINE_AA)
    cv2.line(img_line, pts1[3].astype(np.int64).tolist(), pts1[1].astype(np.int64).tolist(), (0, 0, 255), 2,
             cv2.LINE_AA)
    cv2.line(img_line, pts1[3].astype(np.int64).tolist(), pts1[2].astype(np.int64).tolist(), (0, 0, 255), 2,
             cv2.LINE_AA)
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [img_shape[1], 0], [0, img_shape[0]], [img_shape[1], img_shape[0]]])
    pts = cv2.getPerspectiveTransform(pts1, pts2)  # 生成透视变换矩阵
    correct_image = cv2.warpPerspective(image_np, pts, (img_shape[1], img_shape[0]), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101)  # 进行透视变换
    # 返回处理正确后的内容
    cv2.imshow("image_np", image_np)
    cv2.imshow('img_line', img_line)
    cv2.imshow("correct_image", correct_image)
    cv2.waitKey(0)
