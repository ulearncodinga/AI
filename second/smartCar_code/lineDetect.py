
import cv2
import matplotlib.pyplot as plt
import numpy as np



# 视角切换 直视图变为鸟瞰图
def perspective_transform(image_np):
    """

    :param image:
    :return:
    """
    cv2.imshow('image_np',image_np)


    # 定义原始图像中四个顶点的坐标
    points1 = [[62, image_np.shape[0]], [220, 100], [460, image_np.shape[0]], [280, 100]]

    # 透视变换（鸟瞰图）  寻找roi区域
    # cv2.line(image_rgb, points1[0], points1[1], (255, 0, 0), 2)  # 左车道线的线
    # cv2.line(image_rgb, points1[2], points1[3], (255, 0, 0), 2)  # 右车道线的线

    cv2.imshow('image_rgb',image_np)

    points1 = np.float32(points1)
    # 定义目标图像中，四个顶点的对应位置
    x_offset = 120

    points2 = np.float32([[points1[0][0] + 20, points1[0][1]],
                          [points1[1][0] - x_offset, 0],
                          [points1[2][0] - 20, points1[2][1]],
                          [points1[3][0] + x_offset, 0]
                          ]
                         )

    # 获取透视变换矩阵
    M = cv2.getPerspectiveTransform(points1, points2)

    M_INV = cv2.getPerspectiveTransform(points2, points1)

    """
    进行透视变换
    cv2.warpPerspective 用于 对图像执行透视变换（Perspective Transformation，也叫投影变换）
    透视变换比仿射变换更强大：
        仿射变换保持平行性（平行线还是平行线），但不一定保持长度和角度。
        透视变换甚至能把平行线变成相交线（就像我们肉眼看“铁路轨道延伸到远方会汇聚到一点”）。
    所以 warpPerspective 适合做 图像校正、投影映射、视角变换。  
    dst = cv2.warpPerspective(src, M, dsize, flags=None, borderMode=None, borderValue=None)
    参数和 warpAffine 基本一致，不同的是：
        M 是一个 3×3 透视变换矩阵（而 warpAffine 的 M 是 2×3 仿射矩阵）。
        dsize 是输出图像大小。  
    """
    image_arpPerspective = cv2.warpPerspective(image_np, M, (image_np.shape[1], image_np.shape[0]), flags=cv2.INTER_LANCZOS4)

    return image_arpPerspective,M_INV


# 图像增强
def image_enhance(image_arpPerspective):
    """梯度法提取车道线"""
    image_filter2D = cv2.cvtColor(image_arpPerspective, cv2.COLOR_BGR2GRAY)
    # 进行梯度处理
    image_filter2D = cv2.Sobel(image_filter2D, -1, 1, 0)  # 竖直边缘
    # 高斯滤波
    image_blur = cv2.GaussianBlur(image_filter2D, (5, 5), 1.5)

    # 将大于170像素点设为白色，其余的设为黑色
    ret, image_binary = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY)
    # cv2.imshow('image_binary',image_binary)

    cv2.imshow('image_arpPerspective',image_arpPerspective)
    # 进行闭操作
    img_close = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    #
    # """颜色识别提取车道线"""
    # cv2.imshow('image_arpPerspective',image_arpPerspective)
    # yellow_low = [20, 20, 100]
    # yellow_up = [40, 255, 255]
    # hsv = cv2.cvtColor(image_arpPerspective, cv2.COLOR_BGR2HSV)
    # mask1 = cv2.inRange(hsv, np.array(yellow_low), np.array(yellow_up))
    #
    # white_low = [0, 0, 200]
    # white_up = [180, 30, 255]
    # mask2 = cv2.inRange(hsv, np.array(white_low), np.array(white_up))
    #
    # mask = mask1 | mask2
    #
    # # 进行闭操作
    # img_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
    #
    # cv2.imshow('mask1',mask1)
    # cv2.imshow('mask2', mask2)

    return img_close

def find_stop_line(image_arpPerspective):
    global stop_line_flag
    stop_line_flag = False
    hsv = cv2.cvtColor(image_arpPerspective, cv2.COLOR_BGR2HSV)
    white_low = [0, 0, 200]
    white_up = [180, 30, 255]
    mask = cv2.inRange(hsv, np.array(white_low), np.array(white_up))
    # 进行闭操作
    img_close = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    # Canny边缘检测，得到边缘点
    # image_canny = cv2.Canny(img_close, 30, 70)
    """1、统计概率霍夫直线检测"""
    # # 直接返回 线的起点与终点坐标   minLineLength 最小长度  maxLineGap 最大间隔
    # lines = cv2.HoughLinesP(image_canny, 1, np.pi / 180, 80, minLineLength=90, maxLineGap=10)
    # if lines is not None:
    #     print("线个数：",len(lines))
    #     for line in lines:
    #         x1, y1, x2, y2 = line[0]
    #         cv2.line(image_arpPerspective, (x1, y1), (x2, y2), (0, 255, 0), 2)

    Contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if len(Contours) > 0:
        # 取出面积最大的轮廓
        Contour = max(Contours, key=cv2.contourArea)
        Area  = cv2.contourArea(Contour)
        if Area>2000:
            print('发现车道线，车道线面积：',Area)
            # 通过boundingRect获取当前轮廓点所构成的外接矩形的左上角的点的坐标及外接矩形的宽度和高度
            x, y, w, h = cv2.boundingRect(Contour)
            # 此时的(x,y)就是左上角的点的坐标 w是矩形的宽度 h是矩形的高度
            top_left = (x, y)
            # 右下角的点的坐标
            bottom_right = (x + w, y + h)
            # 通过rectangle去绘制矩形
            # cv2.rectangle(image_arpPerspective, top_left, bottom_right, (255, 0, 0), 1)
            cv2.drawContours(image_arpPerspective, [Contour], -1, (0, 0, 255), 1)

            # 车与停止线的偏差
            offset = 20
            if bottom_right[1] > image_arpPerspective.shape[0] - offset:
                print('停车线已到')
                stop_line_flag = True


    cv2.imshow('image_arpPerspective_stop',image_arpPerspective)
    cv2.imshow('image_binary_stop',img_close)
    return stop_line_flag

#滑动窗口检测车道线 https://blog.csdn.net/xiongqi123123/article/details/148520422?fromshare=blogdetail&sharetype=blogdetail&sharerId=148520422&sharerefer=PC&sharesource=m0_58308891&sharefrom=from_link
def find_line(img_close):
    """寻找左右车道线起始位置"""
    """
       我们通常对图像的底部进行水平方向（X轴）的直方图统计，找到像素值为1（或255）
       的像素在每一列的数量，从而绘制出一个表示白色像素数量随水平位置变化的图像直方图，
       在这幅直方图中出现峰值的位置，往往就是左右车道线起始位置所在的区域。
       
    """
    """切出固定区域 解决右车道线在做车道线大于中点时，右车道线起点定位不准问题 计算右车道线起始点"""
    img_close_roi = img_close[img_close.shape[0] // 2:, :]
    # 由于车是向前走的，在图像中表示为车是向上的，所以通常情况下，我们更关注黑白图像的下半部分
    white_pixel_counts = np.sum(img_close_roi==255, axis=0)
    # 由于是二值化图像，通过img_close == 255可以将值变成True和False，减少计算量
    # white_pixel_counts = np.sum(img_close == 255, axis=0)
    # x_positions = np.arange(img_close.shape[1])
    # plt.subplot(1, 2, 1), plt.imshow(img_close,cmap='gray')
    # plt.subplot(1, 2, 2), plt.plot(x_positions, white_pixel_counts)

    # 获取图像中间位置
    middle_position = img_close.shape[1]//2
    # 获取车道线起始位置
    left_line_start = np.argmax(white_pixel_counts[:middle_position])
    right_line_start = np.argmax(white_pixel_counts[middle_position:]) + middle_position
    # print("车道线起始位置:",left_line_start, right_line_start)

    """应用滑动窗口逐步寻找车道线"""
    height = img_close.shape[0]
    # 滑动窗口参数
    nwindows = 9  # 窗口数量
    window_height = height // nwindows  # 每个窗口的高度
    margin = 50  # 窗口宽度的一半
    minpix = 25  # 重新定位窗口中心所需的最小像素数

    # 初始化当前位置
    leftx_current = left_line_start
    rightx_current = right_line_start
    # 存储车道线像素的索引
    left_lane_inds = []
    right_lane_inds = []
    # 获取所有非零像素的位置
    nonzero = img_close.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])

    # 创建彩色图用以绘制窗口
    out_img = cv2.merge([img_close, img_close, img_close])

    rightx_pre = rightx_current
    leftx_pre = leftx_current
    for window in range(nwindows):
        # 计算窗口边界
        win_y_low = height - (window + 1) * window_height
        win_y_high = height - window * window_height
        # 左车道线窗口边界
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin
        # 右车道线窗口边界
        win_xright_low = rightx_current - margin
        win_xright_high = rightx_current + margin
        # 在输出图像上绘制窗口
        cv2.rectangle(out_img, (win_xleft_low, win_y_low), (win_xleft_high, win_y_high), (0, 255, 0), 2)
        cv2.rectangle(out_img, (win_xright_low, win_y_low), (win_xright_high, win_y_high), (0, 255, 0), 2)
        # 找到窗口内的非零像素的索引
        good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                          (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]

        good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) & (nonzerox >= win_xright_low) & (
                    nonzerox <= win_xright_high)).nonzero()[0]

        # 添加这些索引到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)
        # 如果找到足够的像素，重新计算窗口中心
        if len(good_left_inds) > minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        else:
            if len(good_right_inds) > minpix:
                # 拿到这里面白色像素点所有的坐标
                xs = nonzerox[good_right_inds]
                # 更新leftx_base值
                offset = int(np.mean(xs)) - rightx_pre
                leftx_current = leftx_current + offset

        if len(good_right_inds) > minpix:
                rightx_current = int(np.mean(nonzerox[good_right_inds]))
        else:
            if len(good_left_inds) > minpix:
                # 拿到这里面白色像素点所有的坐标
                xs = nonzerox[good_left_inds]
                # 更新rightx_base值
                offset = int(np.mean(xs)) - leftx_pre
                rightx_current = rightx_current + offset

        # 记录上一次的位置
        rightx_pre = rightx_current
        leftx_pre = leftx_current
    cv2.imshow('out_img', out_img)
    # 连接索引的列表,往横向去拼接，为了后续更方便的提取出这些像素点的x和y的坐标，以便进行车道线的拟合
    left_lane_inds = np.concatenate(left_lane_inds)
    # print("left_lane_inds：",left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    # 提取左侧和右侧车道线像素的位置
    # left_lane_inds 是一个一维数组, 它包含了左侧车道线在滑动窗口中找到的白色像素点的x坐标的索引
    # 通过将这些索引作为索引器应用到 nonzerox数组上，就可以得到相应的左侧车道线的x坐标
    # leftx 包含了左侧车道线白色像素点的x坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    # -------------------------- 新增：数据有效性校验 --------------------------
    # 1. 检查数据量（二次拟合至少需要3个不同的点）
    min_points = 3  # 拟合次数+1（2次拟合需3个点）
    if len(leftx) < min_points or len(rightx) < min_points:
        print(f"数据量不足：左车道线{len(leftx)}个点，右车道线{len(rightx)}个点（需至少{min_points}个）")
        return None

    # 2. 检查数据是否重复（避免所有y坐标相同，导致矩阵不可逆）
    if len(np.unique(lefty)) < min_points or len(np.unique(righty)) < min_points:
        print("数据重复：车道线像素y坐标过于集中，无法拟合")
        return None

    # 有了坐标之后，就要去对左侧和右侧车道线进行多项式拟合，从而得到拟合的车道线
    # np.polyfit() 是numpy中用于进行多项式拟合的函数
    # 他接受三个参数：x y  和 deg
    # x：自变量数组，  y：因变量数组
    # deg：多项式的次数，如果是2   y = ax^2 + b^x + c
    # left_fit里存放的就是 a、b、c的参数，
    left_fit = np.polyfit(lefty, leftx, 2)
    right_fit = np.polyfit(righty, rightx, 2)

    # 使用np.linspace 生成一组均匀分布的数值，用于表示竖直方向上的像素坐标，方便后续的车道线的绘制 第三个参数为样本数量
    ploty = np.linspace(0, img_close.shape[0] - 1, img_close.shape[0]).astype(int)

    # 使用多项式拟合来估计左侧和右侧车道线的x坐标
    # left_fitx 就是左侧拟合出来的车道线
    left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype(int)
    right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype(int)
    # 计算中间车道线的位置
    middle_fitx = (left_fitx + right_fitx) // 2

    # print("ploty",ploty)
    # print("left_fitx",left_fitx)

    # out_img[lefty, leftx] = [255, 0, 0]
    # out_img[righty, rightx] = [0, 0, 255]

    # cv2.imshow('out_img2', out_img)
    # #
    # out_img[ploty, left_fitx] = [255, 0, 0]
    # out_img[ploty, right_fitx] = [0, 0, 255]

    # cv2.imshow('out_img3', out_img)

    return left_fitx, right_fitx, middle_fitx, ploty



def show_line(image,image_arpPerspective,lineLoc,M_INV):
    if lineLoc == None:
        return
    left_fitx, right_fitx, middle_fitx, ploty = lineLoc

    # 组合车道线坐标
    pts_left = np.vstack([left_fitx, ploty]).T
    pts_right = np.vstack([right_fitx, ploty]).T
    pts_middle = np.vstack([middle_fitx, ploty]).T

    # 绘制车道线
    cv2.polylines(image_arpPerspective, np.int32([pts_left]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(image_arpPerspective, np.int32([pts_right]), isClosed=False, color=(202, 124, 0), thickness=15)
    cv2.polylines(image_arpPerspective, np.int32([pts_middle]), isClosed=False, color=(202, 124, 0), thickness=15)

    # 将得到的车道线的像素点根据逆透视变换映射到原始图像中
    newwarp = cv2.warpPerspective(image_arpPerspective, M_INV,
                                  (image_arpPerspective.shape[1], image_arpPerspective.shape[0]))

    # 经过加权融合得到 gamma: 添加到加权和后的标量值，通常用于调整亮度
    result = cv2.addWeighted(image, 1, newwarp, 1, 0)
    cv2.imshow('result', result)

    return pts_middle