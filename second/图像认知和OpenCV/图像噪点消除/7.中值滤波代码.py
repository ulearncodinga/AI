import cv2
image_np = cv2.imread('./lena.png')

image_media = cv2.medianBlur(image_np,3)

#只能用边界复制来获取填充值
# BORDER_REPLICATE处理边框像素
cv2.imshow('image_np',image_np)
cv2.imshow('image_media',image_media)
cv2.waitKey(0)