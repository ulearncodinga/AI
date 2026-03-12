import cv2

image = cv2.imread('./1.jpg')
image = cv2.resize(image,(400,800))
image_gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
# image_gray = cv2.medianBlur(image_gray,3)
ret,image_binary = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)


gauss = cv2.GaussianBlur(image_binary,(5,5),0)


canny = cv2.Canny(gauss,50,150)

cv2.imshow('canny',canny)
cv2.waitKey(0)