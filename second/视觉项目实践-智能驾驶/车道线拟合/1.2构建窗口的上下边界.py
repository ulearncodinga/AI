import matplotlib.pyplot as plt

import mqtt
import cv2
import numpy as np
import base64
import queue

def b64_to_np(image):
    img_data = base64.b64decode(image['image'])
    img_np = np.frombuffer(img_data,dtype=np.uint8)
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)
    return img

def perspective_transform(image):
    image_size = (image.shape[1],image.shape[0])

    src = np.float32(
        [
            [80,image_size[1]],
            [450,image_size[1]],
            [image_size[0]//2+40,image_size[1]//2-20],
            [image_size[0]//2-40,image_size[1]//2-20]
        ]
    )

    dst = np.float32([
        [image_size[0]/4,image_size[1]],
        [image_size[0]*3/4,image_size[1]],
        [image_size[0]*3/4,0],
        [image_size[0]/4,0]
    ])
    M = cv2.getPerspectiveTransform(src,dst)
    image_warp = cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)
    return image_warp


def dilate_erode(image,kernel_size):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    image_dilate = cv2.dilate(image,kernel,iterations=1)
    result_img = cv2.erode(image_dilate,kernel,iterations=1)
    return result_img

def extract_line_gradient(image_warp):
    #使用梯度提取车道线
    img_Gaussian = cv2.GaussianBlur(image_warp,(5,5),sigmaX=1)
    img_gray = cv2.cvtColor(img_Gaussian,cv2.COLOR_BGR2GRAY)
    res = cv2.Sobel(img_gray,-1,1,0)
    ret,image_thresh = cv2.threshold(res,127,255,cv2.THRESH_BINARY)
    res = dilate_erode(image_thresh,7)
    cv2.imshow('Sobel',res)
    return res



#提取白色车道线
def hslSelect(img,thresh=(190,255)):
    hls = cv2.cvtColor(img,cv2.COLOR_BGR2HLS)
    l_channel = hls[:,:,1]
    l_channel = l_channel / np.max(l_channel) * 255
    binary_output = np.zeros_like(l_channel)
    binary_output[(l_channel>thresh[0])&(l_channel < thresh[1])] = 255
    return binary_output

#提取黄色车道线
def labSeclect(img,thresh=(210,225)):
    #右半边不可能有黄色车道线
    img[:,240:,:] = (0,0,0)
    lab =cv2.cvtColor(img,cv2.COLOR_BGR2Lab)
    lab_b = lab[:,:,2]
    if np.max(lab_b) > 100:
        lab_b = lab_b / np.max(lab_b)*255
    binary_output = np.zeros_like(lab_b)
    binary_output[((lab_b > thresh[0])&(lab_b<thresh[1]))] = 255
    return binary_output



def extract_line_color(image_warp):
    #提取白色车道线:HSL 模型
    hlsL_binary = hslSelect(image_warp)
    cv2.imshow('hlsL_binary',hlsL_binary)

    #提取黄色车道线: lab 模型(只提取 黄色 ,蓝色)
    labB_binary = labSeclect(image_warp)
    cv2.imshow('labB_binary',labB_binary)
    # cv2.waitKey(0)

    #将提取到的黄色和白色车道线进行融合
    combined_binary = np.zeros_like(hlsL_binary)
    combined_binary[(hlsL_binary == 255) | (labB_binary == 255)] = 255

    #对融合后的车道线进行先膨胀后腐蚀的操作
    dilate_erode_image = dilate_erode(combined_binary,15)
    cv2.imshow('combined_binary',dilate_erode_image)
    return dilate_erode_image

def fingding_line(dilate_erode_image):
    #由于车是向前走的,在图像中表示为车是向上走的,所以通常情况下,我们更关注黑白图像的下半部分
    histogram = np.sum(dilate_erode_image[dilate_erode_image.shape[0]//2:,:],axis =0)

    #获取直方图的中点位置,也就是图像宽度的一半
    print(histogram.shape)#480
    midpoint = histogram.shape[0]//2

    # 直方图左侧最高点的位置
    leftx_base = np.argmax(histogram[:midpoint])
    # print(leftx_base)

    # 直方图右侧最高点的位置
    rightx_base = np.argmax(histogram[midpoint:]) + midpoint

    #获取图像中所有的非零像素的x和y的位置 返回的是行索引和列索引
    nonzero = dilate_erode_image.nonzero()
    nonzeroy = np.array(nonzero[0])
    nonzerox = np.array(nonzero[1])


    #定义一些小窗口的概念
    #定义小窗口的个数
    nwindows = 10
    window_height = dilate_erode_image.shape[0] // nwindows
    #小窗口的宽度
    margin = 50
    #小窗口内白色像素点的个数阈值
    minpix = 40
    #初始化当前窗口的位置后面会持续更新
    leftx_current = leftx_base
    rightx_current = rightx_base

    #创建空列表接收左侧和右侧车道线像素的索引
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        #计算当前窗口的上边界的y坐标
        win_y_high = dilate_erode_image.shape[0] - (window + 1) * window_height
        #计算当前窗口的下边界的y坐标
        win_y_low = dilate_erode_image.shape[0] - window * window_height






    # print(right_base)
    # plt.plot(histogram)
    # plt.show()




if __name__ == '__main__':

    image = cv2.imread('./2.png')

    image_warp = perspective_transform(image)
    image_warp_copy = image_warp.copy()
    #第一种 使用梯度来提取车道线
    # dilate_erode_image = extract_line_gradient(image_warp)

    #第二种 使用颜色提取车道线
    dilate_erode_image = extract_line_color(image_warp_copy)

    fingding_line(dilate_erode_image)

    cv2.imshow('image',image)
    cv2.imshow('image_warp',image_warp)
    cv2.waitKey(0)