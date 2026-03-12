import mqtt
import queue
import base64
import numpy as np
import cv2

def b64_to_np(image):
    #此时的image还是一个字典,我们需要的是字典的值,字典的值是base64格式的图像数据
    img_data = base64.b64decode(image['image'])
    img_np = np.frombuffer(img_data,dtype =np.uint8)
    img = cv2.imdecode(img_np,cv2.IMREAD_COLOR)
    return img

def perspective_transform(image):
    #                图像宽度       图像高度
    image_size = (image.shape[1],image.shape[0])


    # cv2.line(image,(322,135),(470,270),(0,0,255),1)
    # cv2.line(image,(102,270),(200,135),(0,0,255),1)
    src = np.float32(
        [[80, image_size[1]],
         [450, image_size[1]],
         [image_size[0] // 2 + 40, image_size[1] // 2 - 20],
         [image_size[0] // 2 - 40, image_size[1] // 2 - 20]]
    )

    dst = np.float32(
        [[image_size[0] / 4, image_size[1]],
         [image_size[0] * 3 / 4, image_size[1]],
         [image_size[0] * 3 / 4, 0],
         [image_size[0] / 4, 0]]
    )

    #获取透视变换矩阵(图像矫正)
    M =cv2.getPerspectiveTransform(src,dst)

    #调用函数进行透视变换
    image_warp = cv2.warpPerspective(image,M,image_size,flags=cv2.INTER_LINEAR)

    return image_warp




if __name__ == '__main__':
    image = cv2.imread('./1.png')
    image_warp = perspective_transform(image)



    cv2.imshow('image',image)
    cv2.imshow('image_warp',image_warp)
    cv2.waitKey(0)