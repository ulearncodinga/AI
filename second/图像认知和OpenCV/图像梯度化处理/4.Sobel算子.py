import cv2
image_np = cv2.imread('./shudu.png')


#三种情况
image_Sobel = cv2.Sobel(image_np,-1,1,0)
# image_Sobel = cv2.Sobel(image_np,-1,0,1)
# image_Sobel = cv2.Sobel(image_np,-1,1,1)




cv2.imshow('image_np',image_np)
cv2.imshow('image_Sobel',image_Sobel)

cv2.waitKey(0)