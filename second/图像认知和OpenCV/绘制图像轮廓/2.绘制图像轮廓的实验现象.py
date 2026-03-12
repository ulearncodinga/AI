''''''
'''
8位的单通道图 目标物体为白色   背景物体为黑色

图片输入->灰度化->二值化->寻找轮廓->绘制轮廓->输出图片



EXTR_EXTERNAL模式表示:只列出最外层的轮廓

CHAIN_APPROX_SIMPLE方法表示:
    只存储有用的点,比如直线只存储起点和终点


cv2.findContours()函数的原理基于:
    边界跟踪
    
cv2.findContours的返回值contours表示 轮廓点的列表,每个子列表代表一个轮廓的坐标

cv2.findContours的返回值 hierarchy中 hierarchy[i][0]表示 第i条轮廓的后一条轮廓的索引
cv2.findContours() 函数返回的 hierarchy 是一个数组，每个元素是一个四元组 [next, previous, child, parent]。
这四个值分别表示：
next：下一条轮廓的索引。
previous：上一条轮廓的索引。
child：子轮廓的索引。
parent：父轮廓的索引。



cv2.findContours函数中的mode参数中,RETR_LIST模式的特点是
    列出所有轮廓,轮廓关系中无夫轮廓与子轮廓的索引

寻找轮廓的算法论文中,4连通场景指的是 一个1像素点的上下左右四个方向上存在1像素点

边界跟踪的算法核心思想
    从图像的边界点开始,沿着边界进行跟踪,直到回到起点
'''