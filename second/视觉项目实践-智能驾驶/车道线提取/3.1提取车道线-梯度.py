import cv2
import numpy as np
import base64
import mqtt
import queue

def b64_to_np(image):
    img_data = base64.b64decode(image['image'])
    img_np = np.frombuffer(img_data,dtype=np.uint8)
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)
    return img
def perspective_transform(image):
    image_size = (image.shape[1],image.shape[0])

    src = np.float32(
        [[80,image_size[1]],
         [450,image_size[1]],
         [image_size[0]//2+40,image_size[1]//2-20],
         [image_size[0]//2-40,image_size[1]//2-20]]
    )

    dst =np.float32(
         [
             [image_size[0]/4,image_size[1]],
             [image_size[0]*3/4,image_size[1]],
             [image_size[0]*3/4,0],
             [image_size[0]/4,0]
         ]
    )

    M = cv2.getPerspectiveTransform(src,dst)

    image_warp = cv2.warpPerspective(image,M,image_size,flags = cv2.INTER_LINEAR)
    return image_warp

def dilate_erode(image,kernel_size):
    kernel = np.ones((kernel_size,kernel_size),np.uint8)
    image_dilate = cv2.dilate(image,kernel,iterations=1)
    result_img = cv2.erode(image_dilate,kernel,iterations=1)
    return result_img




def extract_line_gradient(image_warp):
    #使用梯度的概念提取车道线
    #对传进来的鸟瞰图进行一次滤波
    img_Gaussian = cv2.GaussianBlur(image_warp,(5,5),sigmaX=1)
    #进行灰度化
    img_gray = cv2.cvtColor(img_Gaussian,cv2.COLOR_BGR2GRAY)
    #使用Sobel算子进行梯度化
    res = cv2.Sobel(img_gray,-1,1,0)

    #二值化
    ret,img_thresh = cv2.threshold(res,127,255,cv2.THRESH_BINARY)

    res = dilate_erode(img_thresh,7)
    cv2.imshow('Sobel',res)







if __name__ == '__main__':
    image = cv2.imread('./2.png')
    #对图像进行透视变换
    image_warp = perspective_transform(image)
    #提取车道线
    #第一种:使用梯度来提取车道线
    dilate_erode_image = extract_line_gradient(image_warp)




    #第二种:使用颜色来提取车道线(BGR->HSV,分别输入H,S,V范围)
    cv2.imshow('image',image)
    cv2.imshow('image_warp',image_warp)
    cv2.waitKey(0)





















