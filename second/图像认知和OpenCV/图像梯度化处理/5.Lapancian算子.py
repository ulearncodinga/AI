import cv2
image_np = cv2.imread('./shudu.png')
# 卷积
image_Lap = cv2.Laplacian(image_np,-1)

cv2.imshow('image_np',image_np)
cv2.imshow('image_Lap',image_Lap)
cv2.waitKey(0)