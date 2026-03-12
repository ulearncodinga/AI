# 中心像素值附近的两行两列最差得到的两个方向的梯度:水平和垂直


# 重点:构建卷积核

# 水平方向的梯度,提取垂直方向的边缘
# 垂直方向的梯度,提取水平方向的边缘

# cv2.filter2D(src, ddepth, kernel, dst=None, anchor=None, delta=None, borderType=None)
'''参数
src：输入图像（可以是单通道或多通道）
ddepth：输出图像的深度（数据类型），通常用 -1 表示与输入图像深度相同
kernel：卷积核（必须是单通道矩阵，如 3x3、5x5 等）
dst：输出图像（可选）
anchor：核的锚点（卷积时的中心位置），默认 (-1,-1) 表示核中心
delta：额外添加到输出像素的值（可选，通常为 0）
borderType：边界填充方式（处理图像边缘像素，默认 cv2.BORDER_DEFAULT）'''
