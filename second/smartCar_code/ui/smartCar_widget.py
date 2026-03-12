#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
智能驾驶系统界面
使用PyQt5实现的智能驾驶前端界面
包含摄像头显示、功能选择、车辆控制等模块
"""

import sys
import time
import base64
import numpy as np
import cv2
from PyQt5.QtWidgets import *
from PyQt5.QtCore import *
from PyQt5.QtGui import *

# 导入业务逻辑模块
try:
    from hqyj_mqtt import Mqtt_Clt
    # 只导入我们需要的部分，避免cv2.imshow的问题
    import matplotlib.pyplot as plt
    # 导入PID控制器
    from pid import PID
    # 导入YOLO检测功能
    from tools import yolo_detect
    LINEDETECT_AVAILABLE = True
    PID_AVAILABLE = True
    YOLO_AVAILABLE = True
except ImportError as e:
    print(f"警告: 无法导入模块: {e}")
    print("将以模拟模式运行")
    Mqtt_Clt = None
    PID = None
    yolo_detect = None
    LINEDETECT_AVAILABLE = False
    PID_AVAILABLE = False
    YOLO_AVAILABLE = False


class MQTTImageThread(QThread):
    """MQTT图像接收线程类"""
    # 信号定义
    image_received = pyqtSignal(np.ndarray)  # 图像接收信号
    connection_status = pyqtSignal(str)  # 连接状态信号
    error_occurred = pyqtSignal(str)  # 错误信号
    
    def __init__(self):
        super().__init__()
        self.mqtt_client = None
        self.running = False
        self.ip_broker = '127.0.0.1'
        self.port_broker = 21883
        self.topic_sub = 'bb'
        self.topic_pub = 'aa'
        self.time_out_secs = 60
        # 检测功能开关
        self.lane_detection_enabled = False
        self.traffic_light_detection_enabled = False  # 红绿灯检测开关
        self.pedestrian_detection_enabled = False     # 行人检测开关
        # 自动驾驶相关参数
        self.auto_driving_enabled = False
        self.auto_speed = 20  # 默认自动驾驶速度
        self.pid_controller = None
        self.emergency_stop = False  # 紧急停车标志
        self.last_lane_detection_time = 0  # 上次检测到车道线的时间
        self.lane_lost_threshold = 3.0  # 车道线丢失超时阈值（秒）
        # 红绿灯和行人检测状态
        self.current_light_color = None  # 当前红绿灯颜色 0:红 1:黄 2:绿
        self.person_detected = False     # 是否检测到行人
        self.waiting_for_green = False   # 是否在等待绿灯
        self.last_green_start_time = 0   # 上次绿灯启动的时间
        self.green_light_cooldown = 3.0  # 绿灯启动后的冷却时间（秒）
        self.setup_pid_controller()
        
    def set_lane_detection_enabled(self, enabled):
        """设置车道线检测功能开关"""
        self.lane_detection_enabled = enabled
    
    def set_traffic_light_detection_enabled(self, enabled):
        """设置红绿灯检测功能开关"""
        self.traffic_light_detection_enabled = enabled
        print(f"红绿灯检测功能: {'启用' if enabled else '禁用'}")
    
    def set_pedestrian_detection_enabled(self, enabled):
        """设置行人检测功能开关"""
        self.pedestrian_detection_enabled = enabled
        print(f"行人检测功能: {'启用' if enabled else '禁用'}")
    
    def set_auto_driving_enabled(self, enabled):
        """设置自动驾驶功能开关"""
        self.auto_driving_enabled = enabled
        print(f"自动驾驶功能: {'启用' if enabled else '禁用'}")
    
    def set_auto_speed(self, speed):
        """设置自动驾驶速度"""
        self.auto_speed = max(0, min(100, speed))  # 限制速度范围0-100
        print(f"自动驾驶速度设置为: {self.auto_speed}")
    
    def setup_pid_controller(self):
        """设置PID控制器"""
        if PID_AVAILABLE:
            # 参考demo.py中的PID参数设置
            self.pid_controller = PID(
                Kp=0.2,  # 比例系数
                Ki=0.0,  # 积分系数
                Kd=0.0,  # 微分系数
                setpoint=240  # 目标位置（图像中心附近）
            )
            self.pid_controller.sample_time = 0.1  # 采样时间
            self.pid_controller.output_limits = (-13, 13)  # 输出限制（转向角度范围）
            print("PID控制器初始化完成")
        else:
            print("警告: PID模块不可用，自动驾驶功能将受限")
    
    def emergency_stop_vehicle(self, reason="未知原因"):
        """紧急停车"""
        self.emergency_stop = True
        print(f"⚠️ 紧急停车！原因: {reason}")
        if self.mqtt_client:
            self.mqtt_client.control_device('carSpeed', 0)
            self.mqtt_client.control_device('carDirection', 0)
    
    def reset_emergency_stop(self):
        """重置紧急停车状态"""
        self.emergency_stop = False
        print("✅ 紧急停车状态已重置")
        
    def setup_mqtt(self):
        """设置MQTT客户端连接"""
        try:
            if Mqtt_Clt is None:
                self.error_occurred.emit("MQTT模块未安装")
                return False
                
            self.connection_status.emit("连接中")
            self.mqtt_client = Mqtt_Clt(
                self.ip_broker, 
                self.port_broker, 
                self.topic_sub, 
                self.topic_pub, 
                self.time_out_secs
            )
            self.connection_status.emit("已连接")
            return True
        except Exception as e:
            self.error_occurred.emit(f"MQTT连接失败: {str(e)}")
            return False
    
    def start_receiving(self):
        """开始接收图像"""
        if self.setup_mqtt():
            self.running = True
            self.start()
        
    def stop_receiving(self):
        """停止接收图像"""
        self.running = False
        if self.isRunning():
            self.wait()
        self.connection_status.emit("未连接")
    
    def run(self):
        """线程主循环 - 接收和处理MQTT图像数据"""
        if not self.mqtt_client:
            self.error_occurred.emit("MQTT客户端未初始化")
            return
            
        while self.running:
            try:
                # 从MQTT队列获取消息（参考demo.py的实现）
                json_msg = self.mqtt_client.mqtt_queue.get(timeout=1)
                
                # 判断是否为图像数据
                if 'image' in json_msg:
                    # 将base64编码解码为原始的二进制数据
                    image_data = base64.b64decode(json_msg['image'])
                    # 将二进制数据转换为np.uint8类型的数组
                    image_array = np.frombuffer(image_data, np.uint8)
                    # 将numpy数组转换为opencv图像（BGR格式）
                    image_bgr = cv2.imdecode(image_array, cv2.IMREAD_COLOR)
                    
                    if image_bgr is not None:
                        # 处理车道线检测（如果启用）
                        processed_image = self.process_lane_detection(image_bgr.copy())
                        # 发送处理后的图像信号
                        self.image_received.emit(processed_image)
                    
            except Exception as e:
                if self.running:  # 只有在运行时才报错
                    # 处理队列为空的情况（正常现象）
                    if "Empty" not in str(e):
                        self.error_occurred.emit(f"接收图像错误: {str(e)}")
                time.sleep(0.1)  # 避免CPU占用过高
    
    def process_lane_detection(self, image_bgr):
        """处理车道线检测"""
        try:
            # 如果车道线检测未启用或模块不可用，直接返回原图
            if not self.lane_detection_enabled or not LINEDETECT_AVAILABLE:
                return image_bgr
            
            print("开始车道线检测...")  # 调试信息
            
            # 执行车道线检测（自己实现，避免cv2.imshow问题）
            # 1. 透视变换
            image_perspective, M_INV = self.perspective_transform_no_show(image_bgr.copy())
            print(f"透视变换完成，图像尺寸: {image_perspective.shape}")
            
            # 2. 如果启用自动驾驶，检测停车线（为后续交通灯识别做准备）
            stop_line_detected = False
            if self.auto_driving_enabled:
                # 检查是否在绿灯冷却时间内，避免重复触发
                current_time = time.time()
                if current_time - self.last_green_start_time > self.green_light_cooldown:
                    stop_line_detected = self.find_stop_line_no_show(image_perspective.copy())
                    if stop_line_detected:
                        print("检测到停车线（为交通灯识别做准备）")
                else:
                    print(f"绿灯冷却中，剩余时间: {self.green_light_cooldown - (current_time - self.last_green_start_time):.1f}秒")
            
            # 3. 执行YOLO检测（红绿灯和行人）
            light_color, person_warning = self.yolo_detection(image_bgr.copy())
            
            # 4. 行人检测紧急停车处理
            if person_warning and self.pedestrian_detection_enabled and self.auto_driving_enabled:
                self.emergency_stop_vehicle("检测到行人")
                return image_bgr  # 紧急停车时直接返回
            
            # 5. 图像增强
            img_close = self.image_enhance_no_show(image_perspective.copy())
            print(f"图像增强完成，二值化图像尺寸: {img_close.shape}")
            
            # 6. 查找车道线
            lineLoc = self.find_line_no_show(img_close)
            print(f"车道线查找结果: {lineLoc is not None}")
            
            if lineLoc is not None:
                print("检测到车道线，正在绘制...")  # 调试信息
                # 7. 绘制车道线
                result_image = self.draw_lane_lines(image_bgr.copy(), image_perspective.copy(), lineLoc, M_INV)
                
                # 8. 如果启用自动驾驶，执行自动控制（包含红绿灯逻辑）
                if self.auto_driving_enabled:
                    self.auto_drive_control(image_bgr, lineLoc, stop_line_detected, light_color)
                
                return result_image
            else:
                print("未检测到车道线")  # 调试信息
                return image_bgr
                
        except Exception as e:
            print(f"车道线检测错误: {e}")
            return image_bgr
    
    def draw_lane_lines(self, image, image_perspective, lineLoc, M_INV):
        """绘制车道线到原图上"""
        try:
            left_fitx, right_fitx, middle_fitx, ploty = lineLoc
            
            # 创建一个空的图像用于绘制车道线
            lane_img = np.zeros_like(image_perspective)
            
            # 组合车道线坐标
            pts_left = np.vstack([left_fitx, ploty]).T
            pts_right = np.vstack([right_fitx, ploty]).T
            pts_middle = np.vstack([middle_fitx, ploty]).T
            
            # 在透视图上绘制车道线（使用更明显的颜色）
            cv2.polylines(lane_img, np.int32([pts_left]), isClosed=False, color=(0, 255, 0), thickness=8)    # 绿色左车道线
            cv2.polylines(lane_img, np.int32([pts_right]), isClosed=False, color=(0, 255, 0), thickness=8)   # 绿色右车道线  
            cv2.polylines(lane_img, np.int32([pts_middle]), isClosed=False, color=(0, 0, 255), thickness=6)  # 红色中心线
            
            # 将车道线图像转换回原始视角
            newwarp = cv2.warpPerspective(lane_img, M_INV, (image.shape[1], image.shape[0]))
            
            # 将车道线叠加到原图上
            result = cv2.addWeighted(image, 0.8, newwarp, 1.0, 0)
            
            print("车道线绘制完成")  # 调试信息
            return result
            
        except Exception as e:
            print(f"绘制车道线错误: {e}")
            return image
    
    def perspective_transform_no_show(self, image_np):
        """透视变换（不显示窗口版本）"""
        # 定义原始图像中四个顶点的坐标
        points1 = [[62, image_np.shape[0]], [220, 100], [460, image_np.shape[0]], [280, 100]]
        points1 = np.float32(points1)
        
        # 定义目标图像中，四个顶点的对应位置
        x_offset = 120
        points2 = np.float32([[points1[0][0] + 20, points1[0][1]],
                              [points1[1][0] - x_offset, 0],
                              [points1[2][0] - 20, points1[2][1]],
                              [points1[3][0] + x_offset, 0]])
        
        # 获取透视变换矩阵
        M = cv2.getPerspectiveTransform(points1, points2)
        M_INV = cv2.getPerspectiveTransform(points2, points1)
        
        # 进行透视变换
        image_arpPerspective = cv2.warpPerspective(image_np, M, (image_np.shape[1], image_np.shape[0]))
        return image_arpPerspective, M_INV
    
    def image_enhance_no_show(self, image_arpPerspective):
        """图像增强（不显示窗口版本）"""
        image_filter2D = cv2.cvtColor(image_arpPerspective, cv2.COLOR_BGR2GRAY)
        # 进行梯度处理
        image_filter2D = cv2.Sobel(image_filter2D, -1, 1, 0)  # 竖直边缘
        # 高斯滤波
        image_blur = cv2.GaussianBlur(image_filter2D, (5, 5), 1.5)
        # 二值化
        ret, image_binary = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY)
        # 进行闭操作
        img_close = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
        return img_close
    
    def find_line_no_show(self, img_close):
        """查找车道线（不显示窗口版本）"""
        try:
            # 切出固定区域
            img_close_roi = img_close[img_close.shape[0] // 2:, :]
            white_pixel_counts = np.sum(img_close_roi == 255, axis=0)
            
            # 获取图像中间位置
            middle_position = img_close.shape[1] // 2
            # 获取车道线起始位置
            left_line_start = np.argmax(white_pixel_counts[:middle_position])
            right_line_start = np.argmax(white_pixel_counts[middle_position:]) + middle_position
            
            # 应用滑动窗口逐步寻找车道线
            height = img_close.shape[0]
            nwindows = 9  # 窗口数量
            window_height = height // nwindows  # 每个窗口的高度
            margin = 50  # 窗口宽度的一半
            minpix = 25
            
            # 找到所有非零像素的坐标
            nonzero = img_close.nonzero()
            nonzeroy = np.array(nonzero[0])
            nonzerox = np.array(nonzero[1])
            
            # 当前窗口中心的x坐标
            leftx_current = left_line_start
            rightx_current = right_line_start
            
            # 存储车道线像素索引的列表
            left_lane_inds = []
            right_lane_inds = []
            
            # 记录上一次的位置
            leftx_pre = leftx_current
            rightx_pre = rightx_current
            
            # 遍历每个窗口
            for window in range(nwindows):
                # 计算窗口的边界
                win_y_low = img_close.shape[0] - (window + 1) * window_height
                win_y_high = img_close.shape[0] - window * window_height
                win_xleft_low = leftx_current - margin
                win_xleft_high = leftx_current + margin
                win_xright_low = rightx_current - margin
                win_xright_high = rightx_current + margin
                
                # 找到窗口内的非零像素
                good_left_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                  (nonzerox >= win_xleft_low) & (nonzerox < win_xleft_high)).nonzero()[0]
                good_right_inds = ((nonzeroy >= win_y_low) & (nonzeroy < win_y_high) &
                                   (nonzerox >= win_xright_low) & (nonzerox < win_xright_high)).nonzero()[0]
                
                # 将这些索引添加到列表中
                left_lane_inds.append(good_left_inds)
                right_lane_inds.append(good_right_inds)
                
                # 如果找到足够的像素，重新计算窗口中心
                if len(good_left_inds) > minpix:
                    leftx_current = int(np.mean(nonzerox[good_left_inds]))
                else:
                    if len(good_right_inds) > minpix:
                        xs = nonzerox[good_right_inds]
                        offset = int(np.mean(xs)) - rightx_pre
                        leftx_current = leftx_current + offset
                
                if len(good_right_inds) > minpix:
                    rightx_current = int(np.mean(nonzerox[good_right_inds]))
                else:
                    if len(good_left_inds) > minpix:
                        xs = nonzerox[good_left_inds]
                        offset = int(np.mean(xs)) - leftx_pre
                        rightx_current = rightx_current + offset
                
                # 记录上一次的位置
                rightx_pre = rightx_current
                leftx_pre = leftx_current
            
            # 连接索引的列表
            left_lane_inds = np.concatenate(left_lane_inds)
            right_lane_inds = np.concatenate(right_lane_inds)
            
            # 提取左侧和右侧车道线像素的位置
            leftx = nonzerox[left_lane_inds]
            lefty = nonzeroy[left_lane_inds]
            rightx = nonzerox[right_lane_inds]
            righty = nonzeroy[right_lane_inds]
            
            # 数据有效性校验
            min_points = 3
            if len(leftx) < min_points or len(rightx) < min_points:
                print(f"数据量不足：左车道线{len(leftx)}个点，右车道线{len(rightx)}个点")
                return None
            
            if len(np.unique(lefty)) < min_points or len(np.unique(righty)) < min_points:
                print("数据重复：车道线像素y坐标过于集中，无法拟合")
                return None
            
            # 多项式拟合
            left_fit = np.polyfit(lefty, leftx, 2)
            right_fit = np.polyfit(righty, rightx, 2)
            
            # 生成y坐标
            ploty = np.linspace(0, img_close.shape[0] - 1, img_close.shape[0]).astype(int)
            
            # 计算拟合的x坐标
            left_fitx = (left_fit[0] * ploty ** 2 + left_fit[1] * ploty + left_fit[2]).astype(int)
            right_fitx = (right_fit[0] * ploty ** 2 + right_fit[1] * ploty + right_fit[2]).astype(int)
            middle_fitx = (left_fitx + right_fitx) // 2
            
            return left_fitx, right_fitx, middle_fitx, ploty
            
        except Exception as e:
            print(f"查找车道线错误: {e}")
            return None
    
    def find_stop_line_no_show(self, image_perspective):
        """查找停车线（不显示窗口版本）"""
        try:
            # 图像增强处理（类似于原函数）
            image_filter2D = cv2.cvtColor(image_perspective, cv2.COLOR_BGR2GRAY)
            image_filter2D = cv2.Sobel(image_filter2D, -1, 0, 1)  # 水平边缘
            image_blur = cv2.GaussianBlur(image_filter2D, (5, 5), 1.5)
            ret, image_binary = cv2.threshold(image_blur, 120, 255, cv2.THRESH_BINARY)
            img_close = cv2.morphologyEx(image_binary, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))
            
            # 查找轮廓
            contours, hierarchy = cv2.findContours(img_close, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            
            if len(contours) > 0:
                # 取出面积最大的轮廓
                contour = max(contours, key=cv2.contourArea)
                area = cv2.contourArea(contour)
                
                if area > 2000:
                    print(f'发现停车线，面积：{area}')
                    # 获取外接矩形
                    x, y, w, h = cv2.boundingRect(contour)
                    bottom_right = (x + w, y + h)
                    
                    # 车与停止线的偏差检测 - 调整停车距离
                    # 增加停车距离，避免停得太近
                    offset = 60  # 从20增加到60，增加停车距离
                    if bottom_right[1] > image_perspective.shape[0] - offset:
                        print(f'停车线检测到，距离底部: {image_perspective.shape[0] - bottom_right[1]}像素')
                        return True
            
            return False
            
        except Exception as e:
            print(f"停车线检测错误: {e}")
            return False
    
    def yolo_detection(self, image_bgr):
        """YOLO检测红绿灯和行人"""
        try:
            if not YOLO_AVAILABLE:
                return None, False
            
            # 只有在启用相关检测时才执行YOLO
            if not (self.traffic_light_detection_enabled or self.pedestrian_detection_enabled):
                return None, False
            
            print("执行YOLO检测...")
            light_color, person_warning = yolo_detect(image_bgr)
            
            # 更新检测状态
            self.current_light_color = light_color
            self.person_detected = person_warning
            
            if light_color is not None:
                color_names = ["红灯", "黄灯", "绿灯"]
                print(f"检测到红绿灯: {color_names[light_color]}")
            
            if person_warning:
                print("⚠️ 检测到行人，需要紧急停车！")
            
            return light_color, person_warning
            
        except Exception as e:
            print(f"YOLO检测错误: {e}")
            return None, False
    
    def auto_drive_control(self, image_bgr, lineLoc, stop_line_detected=False, light_color=None):
        """自动驾驶控制逻辑（参考demo.py的auto_run函数）"""
        try:
            # 安全检查
            if not self.auto_driving_enabled or self.pid_controller is None or self.mqtt_client is None:
                return
            
            # 紧急停车状态检查
            if self.emergency_stop:
                print("🛑 紧急停车状态，停止控制")
                self.mqtt_client.control_device('carSpeed', 0)
                return
            
            # 红绿灯控制逻辑（严格参考demo.py的实现）
            if stop_line_detected and self.traffic_light_detection_enabled:
                # 首先无条件停车（模拟demo.py的line 72）
                self.mqtt_client.control_device('carSpeed', 0)
                print("🚦 到达停车线，停车检查红绿灯状态")
                
                if light_color is not None:
                    print(f'识别到的颜色为: {light_color}')
                    if light_color == 0 or light_color == 1:  # 红灯或黄灯
                        color_name = "红灯" if light_color == 0 else "黄灯"
                        print(f"🔴 {color_name}，停止车辆")
                        self.mqtt_client.control_device('carSpeed', 0)
                        self.waiting_for_green = True
                        return  # continue等价，不执行后续的auto_run
                    elif light_color == 2:  # 绿灯
                        print("🟢 绿灯，启动车辆")
                        self.mqtt_client.control_device('carSpeed', self.auto_speed)
                        self.waiting_for_green = False
                        # 记录绿灯启动时间，用于冷却
                        self.last_green_start_time = time.time()
                        # 给车辆启动时间（模拟demo.py的time.sleep(2)）
                        time.sleep(0.1)  # 短暂延迟，避免阻塞UI
                        # 继续执行正常的车道线跟随逻辑
                else:
                    # 没有检测到红绿灯但有停车线，保守停车
                    print("⚠️ 检测到停车线但未识别到红绿灯，保守停车")
                    self.mqtt_client.control_device('carSpeed', 0)
                    self.waiting_for_green = True
                    return
            
            # 如果正在等待绿灯，持续检查（类似demo.py的持续循环检查）
            if self.waiting_for_green and self.traffic_light_detection_enabled:
                if light_color == 2:  # 检测到绿灯
                    print("🟢 绿灯亮起，启动车辆")
                    self.mqtt_client.control_device('carSpeed', self.auto_speed)
                    self.waiting_for_green = False
                    # 记录绿灯启动时间，用于冷却
                    self.last_green_start_time = time.time()
                    # 给车辆启动时间
                    time.sleep(0.1)
                    # 继续执行正常的车道线跟随逻辑
                else:
                    print("🔴 仍在等待绿灯")
                    self.mqtt_client.control_device('carSpeed', 0)
                    return  # 不执行后续的auto_run
            
            left_fitx, right_fitx, middle_fitx, ploty = lineLoc
            
            # 更新车道线检测时间
            self.last_lane_detection_time = time.time()
            
            # 计算车道中心位置（参考demo.py中的逻辑）
            # 取图像下半部分的车道中心点的平均值
            pts_middle = np.vstack([middle_fitx, ploty]).T
            
            # 得到车的当前位置x的值（图像中心）
            image_center = image_bgr.shape[1] // 2
            
            # 处理目标位置，得到车道线的x值（取下半部分的平均值）
            # 参考demo.py: land_center = pts_middle[240:,::].mean(axis=0)[0]
            valid_points = pts_middle[240:, :] if len(pts_middle) > 240 else pts_middle
            if len(valid_points) > 0:
                land_center = valid_points.mean(axis=0)[0]
                
                # 安全检查：车道中心偏移过大
                offset = abs(land_center - image_center)
                if offset > image_center * 0.8:  # 偏移超过80%
                    self.emergency_stop_vehicle(f"车道偏移过大: {offset:.1f}")
                    return
                
                # 使用PID控制器计算转向角度
                steering_angle = -self.pid_controller(land_center)
                
                # 安全检查：转向角度限制
                max_steering = 10  # 最大转向角度
                if abs(steering_angle) > max_steering:
                    steering_angle = max_steering if steering_angle > 0 else -max_steering
                    print(f"⚠️ 转向角度被限制到: {steering_angle}")
                
                print(f"自动驾驶控制 - 车道中心: {land_center:.1f}, 图像中心: {image_center}, 转向角: {steering_angle:.2f}")
                
                # 发送控制指令
                self.mqtt_client.control_device('carSpeed', self.auto_speed)
                self.mqtt_client.control_device('carDirection', steering_angle)
                
            else:
                print("警告: 无法获取有效的车道中心点")
                # 如果连续无法获取车道线，执行安全停车
                current_time = time.time()
                if current_time - self.last_lane_detection_time > self.lane_lost_threshold:
                    self.emergency_stop_vehicle("车道线丢失超时")
                
        except Exception as e:
            print(f"自动驾驶控制错误: {e}")
            self.emergency_stop_vehicle(f"控制系统异常: {str(e)}")


class SmartDrivingWidget(QMainWindow):
    """智能驾驶主界面类"""
    
    def __init__(self):
        super().__init__()
        # 初始化MQTT图像接收线程
        self.mqtt_thread = MQTTImageThread()
        self.current_image = None
        
        self.init_ui()
        self.apply_styles()
        self.connect_signals()
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("🚗 智能驾驶系统")
        self.setMinimumSize(1000, 700)
        self.resize(1200, 800)
        
        # 创建中央部件
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 创建主布局 - 仪表盘风格布局
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(10, 10, 10, 10)
        main_layout.setSpacing(8)
        
        # 顶部信息栏
        self.create_top_info_bar(main_layout)
        
        # 主要内容区域 - 中央摄像头 + 周围控制面板
        self.create_dashboard_layout(main_layout)
        
        # 底部状态栏
        self.create_status_bar(main_layout)
        
    def create_top_info_bar(self, layout):
        """创建顶部信息栏"""
        top_bar = QFrame()
        top_bar.setObjectName("topInfoBar")
        top_bar.setFixedHeight(50)
        
        bar_layout = QHBoxLayout(top_bar)
        bar_layout.setContentsMargins(20, 8, 20, 8)
        
        # 左侧：系统标题
        title_label = QLabel("🚗 智能驾驶系统")
        title_label.setObjectName("dashboardTitle")
        bar_layout.addWidget(title_label)
        
        bar_layout.addStretch()
        
        # 中央：智能驾驶状态
        self.smart_mode_indicator = QLabel("⚪ 手动模式")
        self.smart_mode_indicator.setObjectName("modeIndicator")
        bar_layout.addWidget(self.smart_mode_indicator)
        
        bar_layout.addStretch()
        
        # 右侧：时间显示
        self.top_time_label = QLabel("⏰ 00:00:00")
        self.top_time_label.setObjectName("topTimeLabel")
        bar_layout.addWidget(self.top_time_label)
        
        layout.addWidget(top_bar)
        
    def create_dashboard_layout(self, layout):
        """创建仪表盘风格布局 - 中央摄像头，周围控制面板"""
        dashboard_widget = QWidget()
        dashboard_layout = QGridLayout(dashboard_widget)
        dashboard_layout.setContentsMargins(15, 15, 15, 15)
        dashboard_layout.setSpacing(12)
        
        # 左上：功能检测面板
        self.create_detection_panel(dashboard_layout, 0, 0)
        
        # 顶部中央：智能驾驶控制
        self.create_smart_control_panel(dashboard_layout, 0, 1)
        
        # 右上：系统状态面板
        self.create_system_status_panel(dashboard_layout, 0, 2)
        
        # 左中：速度和模式控制
        self.create_speed_control_panel(dashboard_layout, 1, 0)
        
        # 中央：摄像头显示区域（主要区域）
        self.create_central_camera_display(dashboard_layout, 1, 1)
        
        # 右中：车辆控制面板
        self.create_vehicle_control_panel(dashboard_layout, 1, 2)
        
        # 底部跨列：检测状态显示
        self.create_detection_status_bar(dashboard_layout, 2, 0, 1, 3)
        
        # 设置网格布局比例
        dashboard_layout.setRowStretch(0, 1)  # 顶部面板
        dashboard_layout.setRowStretch(1, 4)  # 主要区域
        dashboard_layout.setRowStretch(2, 1)  # 底部状态
        
        dashboard_layout.setColumnStretch(0, 2)  # 左侧面板
        dashboard_layout.setColumnStretch(1, 4)  # 中央摄像头
        dashboard_layout.setColumnStretch(2, 2)  # 右侧面板
        
        layout.addWidget(dashboard_widget, 1)
        
    def create_central_camera_display(self, layout, row, col):
        """创建中央摄像头显示区域"""
        camera_frame = QFrame()
        camera_frame.setObjectName("centralCamera")
        
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setContentsMargins(8, 8, 8, 8)
        camera_layout.setSpacing(5)
        
        # 摄像头标题
        camera_title = QLabel("🎥 前置摄像头")
        camera_title.setObjectName("cameraTitle")
        camera_title.setAlignment(Qt.AlignCenter)
        camera_layout.addWidget(camera_title)
        
        # 摄像头显示区域
        self.camera_display = QLabel()
        self.camera_display.setObjectName("mainCameraDisplay")
        self.camera_display.setMinimumSize(500, 350)
        self.camera_display.setAlignment(Qt.AlignCenter)
        self.camera_display.setText("📷 摄像头离线\n点击启动按钮激活")
        
        camera_layout.addWidget(self.camera_display, 1)
        layout.addWidget(camera_frame, row, col)
        
    def create_detection_panel(self, layout, row, col):
        """创建功能检测面板（左上）"""
        panel = QFrame()
        panel.setObjectName("dashboardPanel")
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        
        # 面板标题
        title = QLabel("🔍 检测功能")
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)
        
        # 功能复选框
        self.lane_checkbox = QCheckBox("🛣️ 车道线")
        self.traffic_light_checkbox = QCheckBox("🚦 红绿灯")
        self.person_checkbox = QCheckBox("🚶 行人")
        self.fatigue_checkbox = QCheckBox("😴 疲劳")
        
        checkboxes = [self.lane_checkbox, self.traffic_light_checkbox, 
                     self.person_checkbox, self.fatigue_checkbox]
        
        for checkbox in checkboxes:
            checkbox.setObjectName("dashboardCheckbox")
            checkbox.stateChanged.connect(self.on_function_changed)
            panel_layout.addWidget(checkbox)
            
        panel_layout.addStretch()
        layout.addWidget(panel, row, col)
        
    def create_smart_control_panel(self, layout, row, col):
        """创建智能驾驶控制面板（顶部中央）"""
        panel = QFrame()
        panel.setObjectName("dashboardPanel")
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        
        # 智能驾驶开关
        switch_layout = QHBoxLayout()
        switch_layout.addStretch()
        
        self.smart_driving_switch = QCheckBox("自动驾驶")
        self.smart_driving_switch.setObjectName("autoPilotSwitch")
        self.smart_driving_switch.stateChanged.connect(self.on_smart_driving_changed)
        switch_layout.addWidget(self.smart_driving_switch)
        
        switch_layout.addStretch()
        panel_layout.addLayout(switch_layout)
        
        # 系统控制按钮
        button_layout = QHBoxLayout()
        
        self.start_camera_btn = QPushButton("启动")
        self.start_camera_btn.setObjectName("dashboardButton")
        self.start_camera_btn.clicked.connect(self.on_start_camera)
        button_layout.addWidget(self.start_camera_btn)
        
        self.stop_system_btn = QPushButton("停止")
        self.stop_system_btn.setObjectName("dashboardButtonRed")
        self.stop_system_btn.clicked.connect(self.on_stop_system)
        self.stop_system_btn.setEnabled(False)
        button_layout.addWidget(self.stop_system_btn)
        
        panel_layout.addLayout(button_layout)
        layout.addWidget(panel, row, col)
        
    def create_system_status_panel(self, layout, row, col):
        """创建系统状态面板（右上）"""
        panel = QFrame()
        panel.setObjectName("dashboardPanel")
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        
        # 面板标题
        title = QLabel("📡 系统状态")
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)
        
        # 状态指示器
        self.camera_status_indicator = QLabel("📷 离线")
        self.camera_status_indicator.setObjectName("statusIndicator")
        self.camera_status_indicator.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.camera_status_indicator)
        
        self.system_status_indicator = QLabel("⚡ 就绪")
        self.system_status_indicator.setObjectName("statusIndicator")
        self.system_status_indicator.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.system_status_indicator)
        
        panel_layout.addStretch()
        layout.addWidget(panel, row, col)
        
    def create_speed_control_panel(self, layout, row, col):
        """创建速度控制面板（左中）"""
        panel = QFrame()
        panel.setObjectName("dashboardPanel")
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        
        # 面板标题
        title = QLabel("🏃 速度控制")
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)
        
        # 速度显示
        self.speed_display = QLabel("50")
        self.speed_display.setObjectName("speedDisplay")
        self.speed_display.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(self.speed_display)
        
        # 速度单位
        speed_unit = QLabel("公里/时")
        speed_unit.setObjectName("speedUnit")
        speed_unit.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(speed_unit)
        
        # 速度滑块
        self.speed_slider = QSlider(Qt.Vertical)
        self.speed_slider.setObjectName("verticalSpeedSlider")
        self.speed_slider.setRange(0, 100)
        self.speed_slider.setValue(50)
        self.speed_slider.valueChanged.connect(self.on_speed_changed)
        panel_layout.addWidget(self.speed_slider, 1)
        
        layout.addWidget(panel, row, col)
        
    def create_vehicle_control_panel(self, layout, row, col):
        """创建车辆控制面板（右中）"""
        panel = QFrame()
        panel.setObjectName("dashboardPanel")
        
        panel_layout = QVBoxLayout(panel)
        panel_layout.setContentsMargins(10, 10, 10, 10)
        panel_layout.setSpacing(8)
        
        # 面板标题
        title = QLabel("🎮 车辆控制")
        title.setObjectName("panelTitle")
        title.setAlignment(Qt.AlignCenter)
        panel_layout.addWidget(title)
        
        # 方向控制按钮
        control_grid = QGridLayout()
        control_grid.setSpacing(5)
        
        # 前进
        self.forward_btn = QPushButton("▲")
        self.forward_btn.setObjectName("controlButton")
        self.forward_btn.setFixedSize(50, 50)
        self.forward_btn.pressed.connect(lambda: self.on_direction_pressed("forward"))
        self.forward_btn.released.connect(lambda: self.on_direction_released("forward"))
        control_grid.addWidget(self.forward_btn, 0, 1)
        
        # 左转
        self.left_btn = QPushButton("◄")
        self.left_btn.setObjectName("controlButton")
        self.left_btn.setFixedSize(50, 50)
        self.left_btn.pressed.connect(lambda: self.on_direction_pressed("left"))
        self.left_btn.released.connect(lambda: self.on_direction_released("left"))
        control_grid.addWidget(self.left_btn, 1, 0)
        
        # 停止
        self.stop_btn = QPushButton("●")
        self.stop_btn.setObjectName("stopControlButton")
        self.stop_btn.setFixedSize(50, 50)
        self.stop_btn.clicked.connect(lambda: self.on_direction_pressed("stop"))
        control_grid.addWidget(self.stop_btn, 1, 1)
        
        # 右转
        self.right_btn = QPushButton("►")
        self.right_btn.setObjectName("controlButton")
        self.right_btn.setFixedSize(50, 50)
        self.right_btn.pressed.connect(lambda: self.on_direction_pressed("right"))
        self.right_btn.released.connect(lambda: self.on_direction_released("right"))
        control_grid.addWidget(self.right_btn, 1, 2)
        
        # 后退
        self.backward_btn = QPushButton("▼")
        self.backward_btn.setObjectName("controlButton")
        self.backward_btn.setFixedSize(50, 50)
        self.backward_btn.pressed.connect(lambda: self.on_direction_pressed("backward"))
        self.backward_btn.released.connect(lambda: self.on_direction_released("backward"))
        control_grid.addWidget(self.backward_btn, 2, 1)
        
        # 居中对齐
        control_widget = QWidget()
        control_widget.setLayout(control_grid)
        panel_layout.addWidget(control_widget, 0, Qt.AlignCenter)
        
        panel_layout.addStretch()
        layout.addWidget(panel, row, col)
        
    def create_detection_status_bar(self, layout, row, col, row_span, col_span):
        """创建检测状态栏（底部）"""
        status_bar = QFrame()
        status_bar.setObjectName("detectionStatusBar")
        
        bar_layout = QHBoxLayout(status_bar)
        bar_layout.setContentsMargins(15, 8, 15, 8)
        bar_layout.setSpacing(20)
        
        # 检测状态指示器
        self.lane_status = QLabel("🛣️ 车道线: 关闭")
        self.lane_status.setObjectName("detectionStatus")
        bar_layout.addWidget(self.lane_status)
        
        self.traffic_light_status = QLabel("🚦 红绿灯: 关闭")
        self.traffic_light_status.setObjectName("detectionStatus")
        bar_layout.addWidget(self.traffic_light_status)
        
        self.person_status = QLabel("🚶 行人: 关闭")
        self.person_status.setObjectName("detectionStatus")
        bar_layout.addWidget(self.person_status)
        
        self.fatigue_status = QLabel("😴 疲劳: 关闭")
        self.fatigue_status.setObjectName("detectionStatus")
        bar_layout.addWidget(self.fatigue_status)
        
        bar_layout.addStretch()
        
        layout.addWidget(status_bar, row, col, row_span, col_span)
        
    def create_status_bar(self, layout):
        """创建底部状态栏"""
        status_frame = QFrame()
        status_frame.setObjectName("bottomStatusBar")
        status_frame.setFixedHeight(40)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(20, 8, 20, 8)
        
        # 系统状态
        self.system_status_label = QLabel("⚡ 系统就绪")
        self.system_status_label.setObjectName("bottomStatusLabel")
        status_layout.addWidget(self.system_status_label)
        
        status_layout.addStretch()
        
        # 连接状态
        self.connection_status_label = QLabel("📡 未连接")
        self.connection_status_label.setObjectName("bottomStatusLabel")
        status_layout.addWidget(self.connection_status_label)
        
        status_layout.addStretch()
        
        # 时间显示
        self.time_label = QLabel("⏰ 00:00:00")
        self.time_label.setObjectName("bottomStatusLabel")
        status_layout.addWidget(self.time_label)
        
        # 定时器更新时间
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        
        layout.addWidget(status_frame)
        
    def connect_signals(self):
        """连接信号和槽"""
        # 连接MQTT线程的信号
        self.mqtt_thread.image_received.connect(self.display_image)
        self.mqtt_thread.connection_status.connect(self.update_connection_status)
        self.mqtt_thread.error_occurred.connect(self.handle_error)
    
    def display_image(self, image_bgr):
        """在界面上显示图像"""
        try:
            # 将BGR转换为RGB
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            
            # 获取摄像头显示区域的尺寸
            display_size = self.camera_display.size()
            display_width = display_size.width()
            display_height = display_size.height()
            
            # 如果尺寸太小，使用默认尺寸
            if display_width < 100 or display_height < 100:
                display_width, display_height = 500, 350
            
            # 调整图像尺寸以适应显示区域
            height, width, channel = image_rgb.shape
            aspect_ratio = width / height
            
            if aspect_ratio > display_width / display_height:
                # 图像较宽，以宽度为准
                new_width = display_width - 20  # 留一些边距
                new_height = int(new_width / aspect_ratio)
            else:
                # 图像较高，以高度为准
                new_height = display_height - 20  # 留一些边距
                new_width = int(new_height * aspect_ratio)
            
            # 调整图像大小
            resized_image = cv2.resize(image_rgb, (new_width, new_height))
            
            # 转换为QImage
            bytes_per_line = 3 * new_width
            q_image = QImage(resized_image.data, new_width, new_height, bytes_per_line, QImage.Format_RGB888)
            
            # 转换为QPixmap并显示
            pixmap = QPixmap.fromImage(q_image)
            self.camera_display.setPixmap(pixmap)
            self.camera_display.setScaledContents(False)  # 保持宽高比
            
            # 保存当前图像
            self.current_image = image_bgr.copy()
            
        except Exception as e:
            print(f"显示图像错误: {e}")
    
    def update_connection_status(self, status):
        """更新连接状态"""
        status_map = {
            "未连接": ("📡 未连接", "color: #888888;"),
            "连接中": ("📡 连接中", "color: #ffaa00;"),
            "已连接": ("📡 已连接", "color: #00ff00;")
        }
        
        if status in status_map:
            text, style = status_map[status]
            self.connection_status_label.setText(text)
            self.connection_status_label.setStyleSheet(style)
            
            # 当连接成功时，更新摄像头状态
            if status == "已连接":
                self.camera_connected()
    
    def handle_error(self, error_message):
        """处理错误信息"""
        print(f"系统错误: {error_message}")
        self.system_status_label.setText(f"❌ 错误: {error_message}")
        
    def apply_styles(self):
        """应用界面样式 - 仪表盘风格"""
        self.setStyleSheet("""
            /* 主窗口样式 - 深色车载风格 */
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 #0a0a0a, stop:0.3 #1a1a1a, stop:0.7 #2a2a2a, stop:1 #1a1a1a);
                color: #ffffff;
            }
            
            /* 顶部信息栏样式 */
            QFrame#topInfoBar {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 150, 255, 0.1), stop:0.5 rgba(0, 200, 255, 0.15), stop:1 rgba(0, 150, 255, 0.1));
                border: 1px solid rgba(0, 200, 255, 0.3);
                border-radius: 8px;
            }
            
            /* 仪表盘标题样式 */
            QLabel#dashboardTitle {
                font-size: 18px;
                font-weight: bold;
                color: #00ccff;
                letter-spacing: 2px;
            }
            
            /* 模式指示器样式 */
            QLabel#modeIndicator {
                font-size: 16px;
                font-weight: bold;
                color: #ffffff;
                background: rgba(50, 50, 50, 0.8);
                border-radius: 15px;
                padding: 5px 15px;
            }
            
            /* 顶部时间标签样式 */
            QLabel#topTimeLabel {
                font-size: 14px;
                color: #cccccc;
                font-family: 'Consolas', monospace;
            }
            
            /* 仪表盘面板样式 */
            QFrame#dashboardPanel {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(40, 40, 40, 0.9), stop:1 rgba(20, 20, 20, 0.9));
                border: 2px solid rgba(0, 150, 255, 0.3);
                border-radius: 12px;
                margin: 2px;
            }
            
            /* 面板标题样式 */
            QLabel#panelTitle {
                font-size: 14px;
                font-weight: bold;
                color: #00ccff;
                letter-spacing: 1px;
                margin: 5px 0;
            }
            
            /* 中央摄像头区域样式 */
            QFrame#centralCamera {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(10, 10, 10, 0.9), stop:1 rgba(30, 30, 30, 0.9));
                border: 3px solid rgba(0, 200, 255, 0.5);
                border-radius: 15px;
            }
            
            /* 摄像头标题样式 */
            QLabel#cameraTitle {
                font-size: 16px;
                font-weight: bold;
                color: #00ffff;
                letter-spacing: 2px;
                margin: 8px 0;
            }
            
            /* 主摄像头显示区域样式 */
            QLabel#mainCameraDisplay {
                border: 2px dashed rgba(0, 200, 255, 0.5);
                border-radius: 10px;
                background: rgba(0, 0, 0, 0.7);
                color: #888888;
                font-size: 18px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            
            /* 仪表盘复选框样式 */
            QCheckBox#dashboardCheckbox {
                color: #ffffff;
                font-size: 12px;
                font-weight: bold;
                spacing: 6px;
            }
            
            QCheckBox#dashboardCheckbox::indicator {
                width: 16px;
                height: 16px;
                border-radius: 8px;
                border: 2px solid #555555;
                background: rgba(0, 0, 0, 0.5);
            }
            
            QCheckBox#dashboardCheckbox::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #00ff00, stop:1 #00aa00);
                border-color: #00ff00;
            }
            
            QCheckBox#dashboardCheckbox::indicator:hover {
                border-color: #00ccff;
            }
            
            /* 自动驾驶开关样式 */
            QCheckBox#autoPilotSwitch {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                letter-spacing: 1px;
            }
            
            QCheckBox#autoPilotSwitch::indicator {
                width: 24px;
                height: 24px;
                border-radius: 12px;
                border: 3px solid #666666;
                background: rgba(100, 100, 100, 0.3);
            }
            
            QCheckBox#autoPilotSwitch::indicator:checked {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #ff8800, stop:1 #ff4400);
                border-color: #ff6600;
            }
            
            /* 仪表盘按钮样式 */
            QPushButton#dashboardButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 200, 100, 0.8), stop:1 rgba(0, 150, 80, 0.8));
                border: 2px solid rgba(0, 255, 150, 0.4);
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
                color: white;
                padding: 8px 12px;
                letter-spacing: 1px;
            }
            
            QPushButton#dashboardButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 220, 120, 0.9), stop:1 rgba(0, 170, 100, 0.9));
                border-color: rgba(0, 255, 150, 0.6);
            }
            
            QPushButton#dashboardButtonRed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 80, 80, 0.8), stop:1 rgba(200, 40, 40, 0.8));
                border: 2px solid rgba(255, 100, 100, 0.4);
                border-radius: 8px;
                font-size: 12px;
                font-weight: bold;
                color: white;
                padding: 8px 12px;
                letter-spacing: 1px;
            }
            
            QPushButton#dashboardButtonRed:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 100, 100, 0.9), stop:1 rgba(220, 60, 60, 0.9));
                border-color: rgba(255, 100, 100, 0.6);
            }
            
            /* 状态指示器样式 */
            QLabel#statusIndicator {
                color: #cccccc;
                font-size: 11px;
                font-weight: bold;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 6px;
                padding: 4px 8px;
                margin: 2px 0;
            }
            
            /* 速度显示样式 */
            QLabel#speedDisplay {
                font-size: 36px;
                font-weight: bold;
                color: #00ffff;
                font-family: 'Arial', sans-serif;
            }
            
            QLabel#speedUnit {
                font-size: 12px;
                color: #888888;
                font-weight: bold;
                letter-spacing: 1px;
            }
            
            /* 垂直滑块样式 */
            QSlider#verticalSpeedSlider::groove:vertical {
                border: 1px solid rgba(0, 200, 255, 0.3);
                width: 10px;
                background: rgba(0, 0, 0, 0.5);
                border-radius: 5px;
            }
            
            QSlider#verticalSpeedSlider::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #00ffff, stop:1 #0088cc);
                border: 2px solid rgba(0, 255, 255, 0.7);
                height: 20px;
                margin: 0 -5px;
                border-radius: 10px;
            }
            
            QSlider#verticalSpeedSlider::sub-page:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 rgba(0, 255, 255, 0.6), stop:1 rgba(0, 150, 200, 0.6));
                border-radius: 5px;
            }
            
            /* 控制按钮样式 */
            QPushButton#controlButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 150, 255, 0.7), stop:1 rgba(0, 100, 200, 0.7));
                border: 2px solid rgba(0, 200, 255, 0.4);
                border-radius: 25px;
                font-size: 18px;
                font-weight: bold;
                color: white;
            }
            
            QPushButton#controlButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 180, 255, 0.8), stop:1 rgba(0, 130, 220, 0.8));
                border-color: rgba(0, 200, 255, 0.6);
            }
            
            QPushButton#controlButton:pressed {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(0, 120, 200, 0.8), stop:1 rgba(0, 80, 150, 0.8));
            }
            
            QPushButton#stopControlButton {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 100, 100, 0.7), stop:1 rgba(200, 50, 50, 0.7));
                border: 2px solid rgba(255, 150, 150, 0.4);
                border-radius: 25px;
                font-size: 18px;
                font-weight: bold;
                color: white;
            }
            
            QPushButton#stopControlButton:hover {
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 rgba(255, 120, 120, 0.8), stop:1 rgba(220, 70, 70, 0.8));
            }
            
            /* 检测状态栏样式 */
            QFrame#detectionStatusBar {
                background: rgba(20, 20, 20, 0.8);
                border: 1px solid rgba(0, 150, 255, 0.3);
                border-radius: 8px;
            }
            
            QLabel#detectionStatus {
                color: #888888;
                font-size: 11px;
                font-weight: bold;
                background: rgba(0, 0, 0, 0.3);
                border-radius: 4px;
                padding: 3px 6px;
                margin: 2px;
            }
            
            /* 底部状态栏样式 */
            QFrame#bottomStatusBar {
                background: rgba(10, 10, 10, 0.9);
                border-top: 1px solid rgba(0, 150, 255, 0.3);
            }
            
            QLabel#bottomStatusLabel {
                color: #cccccc;
                font-size: 11px;
                font-weight: bold;
                font-family: 'Consolas', monospace;
            }
            
            QPushButton:disabled {
                background: rgba(60, 60, 60, 0.3);
                border-color: rgba(100, 100, 100, 0.2);
                color: rgba(255, 255, 255, 0.3);
            }
        """)
    
    # 事件处理方法
    def on_smart_driving_changed(self, state):
        """智能驾驶状态改变"""
        # 更新MQTT线程中的自动驾驶状态
        is_enabled = (state == Qt.Checked)
        self.mqtt_thread.set_auto_driving_enabled(is_enabled)
        
        if state == Qt.Checked:
            # 启用自动驾驶时，自动启用车道线检测
            if hasattr(self, 'lane_checkbox') and not self.lane_checkbox.isChecked():
                self.lane_checkbox.setChecked(True)
                self.on_function_changed()  # 触发车道线检测开关
            
            self.smart_mode_indicator.setText("🟠 智能模式")
            self.smart_mode_indicator.setStyleSheet("color: #ff6600; background: rgba(255, 102, 0, 0.2); border: 1px solid #ff6600;")
            self.system_status_label.setText("🤖 智能驾驶已启动")
            print("智能驾驶模式已启用")
        else:
            self.smart_mode_indicator.setText("⚪ 手动模式")
            self.smart_mode_indicator.setStyleSheet("color: #cccccc; background: rgba(100, 100, 100, 0.2); border: 1px solid #666666;")
            self.system_status_label.setText("⚡ 系统就绪")
            print("切换到手动驾驶模式")
    
    def on_function_changed(self):
        """功能选择改变"""
        sender = self.sender()
        function_name = sender.text()
        
        if sender.isChecked():
            print(f"启用功能: {function_name}")
            # 更新对应的状态显示
            if "车道线" in function_name:
                self.lane_status.setText("🛣️ 车道线: 激活")
                self.lane_status.setStyleSheet("color: #00ff00; background: rgba(0, 255, 0, 0.1);")
                # 启用MQTT线程的车道线检测
                self.mqtt_thread.set_lane_detection_enabled(True)
            elif "红绿灯" in function_name:
                self.traffic_light_status.setText("🚦 红绿灯: 激活")
                self.traffic_light_status.setStyleSheet("color: #00ff00; background: rgba(0, 255, 0, 0.1);")
                # 启用MQTT线程的红绿灯检测
                self.mqtt_thread.set_traffic_light_detection_enabled(True)
            elif "行人" in function_name:
                self.person_status.setText("🚶 行人: 激活")
                self.person_status.setStyleSheet("color: #00ff00; background: rgba(0, 255, 0, 0.1);")
                # 启用MQTT线程的行人检测
                self.mqtt_thread.set_pedestrian_detection_enabled(True)
            elif "疲劳" in function_name:
                self.fatigue_status.setText("😴 疲劳: 激活")
                self.fatigue_status.setStyleSheet("color: #00ff00; background: rgba(0, 255, 0, 0.1);")
        else:
            print(f"禁用功能: {function_name}")
            # 重置对应的状态显示
            if "车道线" in function_name:
                self.lane_status.setText("🛣️ 车道线: 关闭")
                self.lane_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
                # 禁用MQTT线程的车道线检测
                self.mqtt_thread.set_lane_detection_enabled(False)
            elif "红绿灯" in function_name:
                self.traffic_light_status.setText("🚦 红绿灯: 关闭")
                self.traffic_light_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
                # 禁用MQTT线程的红绿灯检测
                self.mqtt_thread.set_traffic_light_detection_enabled(False)
            elif "行人" in function_name:
                self.person_status.setText("🚶 行人: 关闭")
                self.person_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
                # 禁用MQTT线程的行人检测
                self.mqtt_thread.set_pedestrian_detection_enabled(False)
            elif "疲劳" in function_name:
                self.fatigue_status.setText("😴 疲劳: 关闭")
                self.fatigue_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
    
    def on_direction_pressed(self, direction):
        """方向按钮按下"""
        direction_map = {
            "forward": "前进",
            "backward": "后退", 
            "left": "左转",
            "right": "右转",
            "stop": "停止"
        }
        print(f"车辆控制: {direction_map.get(direction, direction)}")
        
        # 更新系统状态
        if direction != "stop":
            self.system_status_label.setText(f"🚗 车辆{direction_map.get(direction, direction)}中")
        else:
            self.system_status_label.setText("⏹️ 车辆停止")
    
    def on_direction_released(self, direction):
        """方向按钮释放"""
        print(f"停止{direction}")
        self.system_status_label.setText("🟢 系统就绪")
    
    def on_speed_changed(self, value):
        """速度滑块改变"""
        self.speed_display.setText(str(value))
        # 更新自动驾驶速度
        self.mqtt_thread.set_auto_speed(value)
        print(f"速度设置: {value} 公里/时")
    
    def on_start_camera(self):
        """启动MQTT连接和图像接收"""
        print("启动MQTT图像接收")
        
        # 重置紧急停车状态
        self.mqtt_thread.reset_emergency_stop()
        
        # 更新UI状态
        self.camera_display.setText("📹 正在启动MQTT连接...\n请稍候...")
        self.camera_status_indicator.setText("📷 连接中")
        self.camera_status_indicator.setStyleSheet("color: #ffaa00; background: rgba(255, 170, 0, 0.2);")
        self.system_status_label.setText("🔄 系统启动中")
        self.start_camera_btn.setEnabled(False)
        self.stop_system_btn.setEnabled(True)
        
        # 启动MQTT图像接收线程
        self.mqtt_thread.start_receiving()
    
    def camera_connected(self):
        """摄像头连接成功（由MQTT线程信号触发）"""
        self.camera_status_indicator.setText("📷 在线")
        self.camera_status_indicator.setStyleSheet("color: #00ff00; background: rgba(0, 255, 0, 0.2);")
        self.system_status_label.setText("⚡ 系统运行中")
        print("MQTT连接成功，开始接收图像")
    
    def on_stop_system(self):
        """停止系统"""
        print("停止MQTT图像接收")
        
        # 紧急停车
        self.mqtt_thread.emergency_stop_vehicle("用户手动停止系统")
        
        # 停止MQTT线程
        self.mqtt_thread.stop_receiving()
        
        # 重置UI显示
        self.camera_display.clear()  # 清除图像
        self.camera_display.setText("📷 摄像头离线\n点击启动按钮激活")
        self.camera_display.setStyleSheet("")  # 重置为默认样式
        
        self.camera_status_indicator.setText("📷 离线")
        self.camera_status_indicator.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
        self.system_status_label.setText("⚡ 系统就绪")
        
        # 重置自动驾驶状态
        if hasattr(self, 'smart_driving_checkbox'):
            self.smart_driving_checkbox.setChecked(False)
            self.on_smart_driving_changed(Qt.Unchecked)
        self.start_camera_btn.setEnabled(True)
        self.stop_system_btn.setEnabled(False)
        
        # 重置所有检测状态
        self.lane_status.setText("🛣️ 车道线: 关闭")
        self.lane_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
        self.traffic_light_status.setText("🚦 红绿灯: 关闭")
        self.traffic_light_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
        self.person_status.setText("🚶 行人: 关闭")
        self.person_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
        self.fatigue_status.setText("😴 疲劳: 关闭")
        self.fatigue_status.setStyleSheet("color: #888888; background: rgba(0, 0, 0, 0.3);")
        
        # 清空当前图像
        self.current_image = None
    
    def update_time(self):
        """更新时间显示"""
        current_time = QTime.currentTime()
        time_text = current_time.toString("⏰ hh:mm:ss")
        self.time_label.setText(time_text)
        self.top_time_label.setText(time_text)


def main():
    """主函数"""
    app = QApplication(sys.argv)
    
    # 设置应用程序属性
    app.setApplicationName("智能驾驶系统")
    app.setApplicationVersion("1.0")
    
    # 创建主窗口
    window = SmartDrivingWidget()
    window.show()
    
    # 运行应用程序
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
