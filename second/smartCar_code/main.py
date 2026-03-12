import cv2
from matplotlib import  pyplot as plt
from lineDetect import *


if __name__ == '__main__':

    image = cv2.imread('image13.png')
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    # 透视变换
    image_arpPerspective,M_INV = perspective_transform(image,image_rgb)

    # 图像增强
    img_close = image_enhance(image_arpPerspective.copy())
    # cv2.imshow('img_close', img_close)
    # 寻找车道线
    lineLoc =  find_line(img_close)
    # 寻找停止线
    print(find_stop_line(image_arpPerspective.copy()))
    # show_line(image,image_arpPerspective,lineLoc,M_INV)

    # 结果显示

    # plt.subplot(231), plt.imshow(image_rgb),plt.title('original')
    # plt.subplot(232), plt.imshow(image_arpPerspective,cmap='gray'),plt.title('wrap')
    # plt.subplot(233), plt.imshow(image_Contours),plt.title('image_Contours')
    # plt.subplot(234), plt.imshow(img_close,cmap='gray'),plt.title('img_close')
    # plt.show()
    cv2.waitKey(0)