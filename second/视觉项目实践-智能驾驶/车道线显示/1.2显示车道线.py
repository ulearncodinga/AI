import time
import json
import matplotlib.pyplot as plt
from pid import PID
import mqtt
import cv2
import numpy as np
import base64
import queue


class LaneCenterPlotter:
    def __init__(self,max_frames=200,image_height=480):
        #设置matplotlib为交互模式
        plt.ion()
        self.fig,self.ax = plt.subplots()
        self.line_lane_center, = self.ax.plot([],[],'r-',label='Lane Center')
        self.line_image_center, = self.ax.plot([],[],'b-',label='Image Center')

        #设置图标,标题和标签
        self.ax.set_title('Lane and Image Center')
        self.ax.set_xlabel('Frame')
        self.ax.set_ylabel('Pixel Coordinate')
        self.ax.legend()

        #初始化数据
        self.x_data = []
        self.y_data_lane_center = []
        self.y_data_image_center = []
        self.max_frames = max_frames
        self.image_height = image_height

        #初始化折线图

        self.init_plot()


    def init_plot(self):
        self.ax.set_xlim(0,self.max_frames)
        self.ax.set_ylim(0,self.image_height)
        self.line_lane_center.set_data([],[])
        self.line_image_center.set_data([],[])
        self.ax.grid()

    def update_plot(self,frame,lane_center,image_center):
        self.x_data.append(frame)
        self.y_data_lane_center.append(lane_center)
        self.y_data_image_center.append(image_center)

        #更新折线图
        self.line_lane_center.set_data(self.x_data,self.y_data_lane_center)
        self.line_image_center.set_data(self.x_data,self.y_data_image_center)

        #保持x轴的范围固定
        if len(self.x_data) > self.max_frames:
            self.ax.set_xlim(self.x_data[-self.max_frames],self.x_data[-1])
            self.ax.figure.canvas.draw()
        plt.pause(0.01)





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
    #获取透视变换矩阵
    M = cv2.getPerspectiveTransform(src,dst)

    #获取逆透视变换的矩阵
    minv = cv2.getPerspectiveTransform(dst,src)


    #调用函数进行透视变换
    image_warp = cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)
    return image_warp,minv


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
    res = dilate_erode(image_thresh,15)
    # cv2.imshow('Sobel',res)
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
    # cv2.imshow('hlsL_binary',hlsL_binary)

    #提取黄色车道线: lab 模型(只提取 黄色 ,蓝色)
    labB_binary = labSeclect(image_warp)
    # cv2.imshow('labB_binary',labB_binary)
    # cv2.waitKey(0)

    #将提取到的黄色和白色车道线进行融合
    combined_binary = np.zeros_like(hlsL_binary)
    combined_binary[(hlsL_binary == 255) | (labB_binary == 255)] = 255

    #对融合后的车道线进行先膨胀后腐蚀的操作
    dilate_erode_image = dilate_erode(combined_binary,15)
    # cv2.imshow('combined_binary',dilate_erode_image)
    return dilate_erode_image

def finding_line(dilate_erode_image):
    #由于车是向前走的,在图像中表示为车是向上走的,所以通常情况下,我们更关注黑白图像的下半部分
    histogram = np.sum(dilate_erode_image[dilate_erode_image.shape[0]//2:,:],axis =0)

    #创建一个三通道图像,用来显示小窗口寻找车道线的过程
    out_img = np.dstack((dilate_erode_image,dilate_erode_image,dilate_erode_image))
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

    leftx_pre = leftx_current
    rightx_pre = rightx_current

    #创建空列表接收左侧和右侧车道线像素的索引
    left_lane_inds = []
    right_lane_inds = []

    for window in range(nwindows):
        #计算当前窗口的上边界的y坐标
        win_y_high = dilate_erode_image.shape[0] - (window + 1) * window_height
        #计算当前窗口的下边界的y坐标
        win_y_low = dilate_erode_image.shape[0] - window * window_height


        #计算左边窗口左右边界的x坐标
        win_xleft_low = leftx_current - margin
        win_xleft_high = leftx_current + margin


        #计算右边窗口的左右边界的x坐标
        win_xright_low= rightx_current - margin
        win_xright_high = rightx_current + margin


        #在out_img里去显示小窗口的过程
        # cv2.rectangle(out_img,(win_xleft_low,win_y_high),(win_xleft_high,win_y_low),(0,255,0),2)#绘制矩形
        # cv2.rectangle(out_img,(win_xright_low,win_y_high),(win_xright_high,win_y_low),(255,0,0),2)
        # cv2.imshow('out_img',out_img)

        #找到处于窗口内非零像素的索引
        good_left_inds = ((nonzeroy>win_y_high)&(nonzeroy<win_y_low)&(nonzerox>=win_xleft_low)&(nonzerox<win_xleft_high)).nonzero()[0]
        good_right_inds = ((nonzeroy>=win_y_high)&(nonzeroy<win_y_low)&(nonzerox>=win_xright_low)&(nonzerox<win_xright_high)).nonzero()[0]
        # 将获取到的白色像素点的索引添加到列表中
        left_lane_inds.append(good_left_inds)
        right_lane_inds.append(good_right_inds)

        # 更新小窗口位置
        if len(good_left_inds)>minpix:
            leftx_current = int(np.mean(nonzerox[good_left_inds]))
        else:
            if len(good_right_inds)>minpix:
                offest = int(np.mean(nonzerox[good_right_inds])) - rightx_current
                leftx_current = leftx_current + offest

        if len(good_right_inds)>minpix:
            rightx_current = int(np.mean(nonzerox[good_right_inds]))
        else:
            if len(good_left_inds)>minpix:
                offest = int(np.mean(nonzerox[good_left_inds])) - leftx_current
                rightx_current = rightx_current + offest


        leftx_pre = leftx_current
        rightx_pre = rightx_current


        # 连接索引的列表,为了后续更方便的提取出这些像素点的x,y坐标,以便于进行车道线拟合
    left_lane_inds = np.concatenate(left_lane_inds)
    right_lane_inds = np.concatenate(right_lane_inds)

    #提取左侧和右侧车道线的像素位置
    #left_lane_inds是一个一维数组,它包含了左侧车道线在滑动窗口中找到的白色像素点的x坐标的索引
    #通过将这些索引作为索引器应用到nonzerox数组上,就可以得到相应的左侧车道线的x坐标
    #leftx 包含了左侧车道线白色像素点的x坐标
    leftx = nonzerox[left_lane_inds]
    lefty = nonzeroy[left_lane_inds]
    rightx = nonzerox[right_lane_inds]
    righty = nonzeroy[right_lane_inds]


    #有了坐标后就对左侧和右侧车道进行多项式拟合,从而得到拟合的车道线
    # np.polyfit()是Numpy中进行多项式拟合的函数
    #它接受三个参数 x,y,deg
    #x:自变量数组 y:因变量数组 deg:多项式的次数,如果是2,y=ax^2+b^x+c
    #此时left_fit里存放的就是a,b,c的参数
    left_fit = np.polyfit(lefty,leftx,2)
    right_fit = np.polyfit(righty,rightx,2)

    #创建一个模板,用来画图
    #使用np.linspace 生成一组均匀分布的数值,用于表示竖直方向上的像素坐标,方便后续的车道线的绘制
    ploty = np.linspace(0,dilate_erode_image.shape[0]-1,dilate_erode_image.shape[0])

    #使用多项式拟合来估计左侧和右侧车道线的x坐标
    #left_fitx就是左侧拟合出来的车道线
    left_fitx = left_fit[0] * ploty ** 2 + left_fit[1] *ploty + left_fit[2]
    right_fitx = right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]


    #计算中间车道线的位置
    middle_fitx = (left_fitx + right_fitx) // 2

    #使用不同颜色将车道线标出来
    out_img[lefty,leftx] = [255,0,0]
    out_img[righty,rightx] = [0,0,255]

    # cv2.imshow('out_img',out_img)

    return left_fitx,right_fitx,middle_fitx,ploty

    # print(right_base)
    # plt.plot(histogram)
    # plt.show()


def show_line(image,image_warp,dilate_erode_image,minv,left_fitx,right_fitx,middle_fitx,ploty):
    #目标:显示原始图像,显示透视变换后的结果,显示车道线在原始图像的结果,显示单纯的车道线(灰色背景的图,图中只有两个车道线)
    #1.创建一个空白图像,用于在上面绘制检测到的车道线
    warp_zero = np.zeros_like(dilate_erode_image).astype(np.uint8)
    color_warp = np.dstack((warp_zero,warp_zero,warp_zero))

    #2.左侧车道线的所有的x坐标和y坐标
    #组合车道线坐标
    # print(left_fitx.shape)
    # print(right_fitx.shape)
    # print(ploty.shape)
    pts_left = np.transpose(np.vstack([left_fitx,ploty]))
    # print(pts_left)
    pts_right = np.transpose(np.vstack([right_fitx,ploty]))
    pts_middle = np.transpose(np.vstack([middle_fitx,ploty]))

    #绘制车道线
    cv2.polylines(color_warp,np.int32([pts_left]),isClosed=False,color=(202,124,0),thickness=15)
    cv2.polylines(color_warp,np.int32([pts_right]),isClosed=False,color=(202,124,0),thickness=15)
    cv2.polylines(color_warp,np.int32([pts_middle]),isClosed=False,color=(202,124,0),thickness=15)
    # cv2.imshow('color_warp',color_warp)

    #将得到的车道线的像素点根据逆透视变换映射到原始图像中
    newwarp = cv2.warpPerspective(color_warp,minv,(image.shape[1],image.shape[0]))
    # cv2.imshow('newwarp',newwarp)

    #将逆透视变换后的结果和原始图像进行融合
    result1 = cv2.addWeighted(image,1,newwarp,1,0)
    # cv2.imshow('result',result)

    #创建一个灰色图
    background_zero = np.zeros_like(image).astype(np.uint8) + 127

    #经过加权融合得到
    result = cv2.addWeighted(background_zero,1,newwarp,1,0)
    # cv2.imshow('image_warp',image_warp)
    # cv2.imshow('result',result)
    # cv2.imshow('image',image)
    # cv2.imshow('result1',result1)

    concatenate_image = np.concatenate((image,image_warp,result1,result),axis = 1)
    cv2.imshow('concatenate',concatenate_image)
    cv2.waitKey(1)
    return pts_middle


def auto_run(image,mqtt_client,pts_middle,pid,carspeed=20):
    #计算车道中心的像素坐标,目标位置(其实是当前位置,因为在本案例中目标位置和当前位置互换了)
    lane_center = pts_middle[240:,:].mean()

    #当前位置(其实是目标位置)
    image_center = image.shape[1] // 2
    steering_angle = -pid(lane_center)

    #发送指令
    mqtt_client.send_mqtt(json.dumps({"carSpeed":carspeed}))
    mqtt_client.send_mqtt(json.dumps({"carDirection":steering_angle}))
    print('steering_angle',steering_angle)

    return lane_center, image_center











if __name__ == '__main__':

    q_mqtt_data = queue.Queue(5)
    plotter = LaneCenterPlotter()
    frame = 0

    # i = 1
    #1.构建mqtt客户端,连接mqtt服务器,方便和3D场景通信
    mqtt_client = mqtt.MQTTClient('127.0.0.1',21883,'bb','aa',q_mqtt_data)

    frame_count = 0
    start_time = time.time()

    pid=PID(Kp=0.35,Ki=0.1,Kd=0.09,setpoint=240)
    pid.sample_time = 0.1
    pid.output_limits = (-13,13)

    while True:
        try:
            image = q_mqtt_data.get()

            if 'image' in image:





                frame_count += 1
                current_time = time.time()
                elapsed_time = current_time - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0
                print(f'fps:::::{fps:.2f}')



            # if 'image' in image:
                image = b64_to_np(image)

                # image = cv2.imread('./2.png')

                #对图像进行透视变换
                image_warp,minv = perspective_transform(image)
                # image_warp_copy = image_warp.copy()
                #第一种 使用梯度来提取车道线
                dilate_erode_image = extract_line_gradient(image_warp)

                #第二种 使用颜色提取车道线
                # dilate_erode_image = extract_line_color(image_warp_copy)

                #拟合车道线
                left_fitx,right_fitx,middle_fitx,ploty = finding_line(dilate_erode_image)

                #绘制车道线
                pts_middle = show_line(image,image_warp,dilate_erode_image,minv,left_fitx,right_fitx,middle_fitx,ploty)


                #自动驾驶
                lane_center,image_center = auto_run(image,mqtt_client,pts_middle,pid)


                #实时显示误差情况
                plotter.update_plot(frame,lane_center,image_center)
                frame += 1
                # cv2.imshow('image',image)
                # cv2.imshow('image_warp',image_warp)
                # cv2.waitKey(0)
        except Exception as e:
            print(e)