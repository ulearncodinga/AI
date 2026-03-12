# 导入opencv的库（计算机视觉处理核心库）
import cv2

# 1.读取图片（BGR格式，OpenCV默认读取方式）
image_np = cv2.imread('./picture.png')  # 替换为自己的图片路径

# 2.灰度化（降维：3通道彩色图→1通道灰度图，减少计算量）
image_gray = cv2.cvtColor(image_np,cv2.COLOR_BGR2GRAY)

# 3.二值化（黑白分割：将灰度图转为只有0/255的黑白图，突出轮廓）
ret,image_thresh = cv2.threshold(image_gray,127,255,cv2.THRESH_BINARY)

# 4.获取轮廓点的坐标（提取物体边界的关键步骤）
contours,hierarchy = cv2.findContours(image_thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)

# 5.寻找凸包（核心：从轮廓点中筛选出凸包点）
# cv2.convexHull(points,hull,clockwise,returnPoints)
cnt = contours[0]  # 取第一个轮廓（假设图片只有一个物体）
hull = cv2.convexHull(cnt)  # 计算该轮廓的凸包
print(type(hull))  # 输出：<class 'numpy.ndarray'>，凸包点是多维数组

# 6.绘制凸包（把凸包轮廓画在原图上）
# polylines(img,pts,isClosed,color,thickness,lineType,shift)
image_poly = cv2.polylines(image_np,[hull],True,(0,0,255),2)  # 红色、线宽2

# 7.结果显示（弹出窗口展示图片）
cv2.imshow('image_np',image_np)  # 原图+凸包
cv2.imshow('image_poly',image_poly)  # 同image_np（因为polylines直接修改原图）
cv2.waitKey(0)  # 等待按键关闭窗口
cv2.destroyAllWindows()  # 补充：建议加这行，避免窗口残留