import sys
import cv2
import os
import numpy as np
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout, 
                             QHBoxLayout, QLabel, QPushButton, QFrame, QGridLayout,
                             QProgressBar, QTextEdit, QSizePolicy, QMessageBox)
from PyQt5.QtCore import Qt, QTimer, QPropertyAnimation, QEasingCurve, pyqtProperty, QThread, pyqtSignal
from PyQt5.QtGui import QFont, QPixmap, QPainter, QBrush, QColor, QPen, QImage

class FaceRecognitionThread(QThread):
    """人脸识别工作线程"""
    # 信号定义
    frame_ready = pyqtSignal(np.ndarray)  # 发送处理后的帧
    recognition_result = pyqtSignal(str, float)  # 发送识别结果 (姓名, 置信度)
    status_update = pyqtSignal(str)  # 发送状态更新
    error_occurred = pyqtSignal(str)  # 发送错误信息
    
    def __init__(self):
        super().__init__()
        self.is_running = False
        self.cap = None
        self.face_cascade = None
        self.recognizer = None
        self.id_name_dict = {}
        
    def init_recognition(self):
        """初始化人脸识别器"""
        try:
            # 加载人脸检测器
            self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            
            # 检查faces文件夹是否存在
            if not os.path.exists('faces'):
                self.error_occurred.emit("未找到faces文件夹，请先进行人脸录入")
                return False
                
            # 创建识别器
            self.recognizer = cv2.face.LBPHFaceRecognizer_create()
            
            # 加载训练数据
            faces = []
            labels = []
            label_id = 0
            
            for person_name in os.listdir('faces'):
                person_dir = os.path.join('faces', person_name)
                if not os.path.isdir(person_dir):
                    continue
                    
                self.id_name_dict[label_id] = person_name
                
                for img_name in os.listdir(person_dir):
                    if img_name.endswith(('.png', '.jpg', '.jpeg')):
                        img_path = os.path.join(person_dir, img_name)
                        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                        if img is not None:
                            faces.append(img)
                            labels.append(label_id)
                            
                label_id += 1
            
            if len(faces) == 0:
                self.error_occurred.emit("未找到训练数据，请先进行人脸录入")
                return False
                
            # 训练识别器
            self.recognizer.train(faces, np.array(labels))
            self.status_update.emit("人脸识别器初始化完成")
            return True
            
        except Exception as e:
            self.error_occurred.emit(f"初始化失败: {str(e)}")
            return False
    
    def start_recognition(self):
        """开始识别"""
        self.is_running = True
        self.start()
        
    def stop_recognition(self):
        """停止识别"""
        self.is_running = False
        if self.cap:
            self.cap.release()
        self.wait()
        
    def run(self):
        """主运行循环"""
        try:
            # 初始化识别器
            if not self.init_recognition():
                return
                
            # 打开摄像头
            self.cap = cv2.VideoCapture(0)
            if not self.cap.isOpened():
                self.error_occurred.emit("无法打开摄像头")
                return
                
            self.status_update.emit("摄像头已连接，开始识别")
            
            while self.is_running:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                    
                # 转换为灰度图
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                
                # 检测人脸
                faces = self.face_cascade.detectMultiScale(gray, 1.3, 5)
                
                # 处理检测到的人脸
                for (x, y, w, h) in faces:
                    # 绘制人脸框
                    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                    
                    # 提取人脸区域
                    face_roi = gray[y:y + h, x:x + w]
                    
                    # 进行识别
                    if self.recognizer:
                        label, confidence = self.recognizer.predict(face_roi)
                        
                        if confidence < 100:  # 识别成功
                            person_name = self.id_name_dict.get(label, "Unknown")
                            # 在图像上显示结果
                            cv2.putText(frame, f'{person_name}: {confidence:.1f}', 
                                      (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                            # 发送识别结果
                            self.recognition_result.emit(person_name, confidence)
                        else:
                            cv2.putText(frame, 'Unknown', (x, y - 10), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
                            self.recognition_result.emit("Unknown", confidence)
                
                # 发送处理后的帧
                self.frame_ready.emit(frame)
                
                # 控制帧率
                self.msleep(30)  # 约30fps
                
        except Exception as e:
            self.error_occurred.emit(f"识别过程出错: {str(e)}")
        finally:
            if self.cap:
                self.cap.release()

class FaceRecognitionWidget(QMainWindow):
    def __init__(self):
        super().__init__()
        # 初始化识别线程
        self.recognition_thread = FaceRecognitionThread()
        self.setup_signals()
        
        self.init_ui()
        self.setup_animations()
        
    def setup_signals(self):
        """设置信号连接"""
        self.recognition_thread.frame_ready.connect(self.update_camera_display)
        self.recognition_thread.recognition_result.connect(self.update_recognition_result)
        self.recognition_thread.status_update.connect(self.update_status)
        self.recognition_thread.error_occurred.connect(self.handle_error)
        
    def init_ui(self):
        """初始化用户界面"""
        self.setWindowTitle("人脸识别系统 - Face Recognition System")
        
        # 设置初始大小和最小大小，但允许用户调整
        self.resize(800, 600)
        self.setMinimumSize(800, 600)
        
        # 设置主窗口样式
        self.setStyleSheet("""
            QMainWindow {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #1a1a2e, stop:0.5 #16213e, stop:1 #0f3460);
            }
        """)
        
        # 创建中央widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        
        # 主布局 - 针对小窗口减少边距
        main_layout = QVBoxLayout(central_widget)
        main_layout.setSpacing(10)  # 减少间距
        main_layout.setContentsMargins(15, 15, 15, 15)  # 减少边距
        
        # 创建顶部标题区域
        self.create_header(main_layout)
        
        # 创建主要内容区域 - 减少间距
        content_layout = QHBoxLayout()
        content_layout.setSpacing(10)  # 减少间距
        main_layout.addLayout(content_layout)
        
        # 左侧摄像头区域
        self.create_camera_section(content_layout)
        
        # 右侧控制面板
        self.create_control_panel(content_layout)
        
        # 底部状态栏
        self.create_status_bar(main_layout)
        
    def create_header(self, parent_layout):
        """创建顶部标题区域"""
        header_frame = QFrame()
        # 针对小窗口优化标题高度
        header_frame.setMinimumHeight(45)  # 减少最小高度
        header_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        header_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.1);
                border-radius: 15px;
                border: 2px solid rgba(0, 255, 255, 0.3);
            }
        """)
        
        header_layout = QHBoxLayout(header_frame)
        header_layout.setContentsMargins(15, 10, 15, 10)  # 减少内边距
        
        # 标题 - 使用相对字体大小
        title_label = QLabel("🤖 智能人脸识别系统")
        title_label.setStyleSheet("""
            QLabel {
                color: #00ffff;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        
        header_layout.addWidget(title_label)
        parent_layout.addWidget(header_frame)
        
    def create_camera_section(self, parent_layout):
        """创建摄像头预览区域"""
        camera_frame = QFrame()
        camera_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        camera_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.7);
                border-radius: 20px;
                border: 3px solid rgba(0, 255, 255, 0.5);
            }
        """)
        
        camera_layout = QVBoxLayout(camera_frame)
        camera_layout.setSpacing(10)  # 减少间距
        camera_layout.setContentsMargins(15, 15, 15, 15)  # 减少边距
        
        # 摄像头预览标签
        preview_label = QLabel("📹 摄像头预览")
        preview_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        preview_label.setAlignment(Qt.AlignCenter)
        preview_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        camera_layout.addWidget(preview_label)
        
        # 摄像头显示区域 - 完全响应式
        self.camera_display = QLabel()
        self.camera_display.setMinimumSize(300, 200)  # 针对小窗口减少最小尺寸
        self.camera_display.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.camera_display.setScaledContents(True)  # 自动缩放内容
        self.camera_display.setStyleSheet("""
            QLabel {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #2c3e50, stop:0.5 #34495e, stop:1 #2c3e50);
                border: 2px solid rgba(0, 255, 255, 0.3);
                border-radius: 15px;
                color: #7f8c8d;
                font-size: 16px;
            }
        """)
        self.camera_display.setAlignment(Qt.AlignCenter)
        self.camera_display.setText("📷\n\n摄像头未启动\n\n点击'开始识别'按钮启动摄像头")
        camera_layout.addWidget(self.camera_display, stretch=1)
        
        # 识别结果显示区域 - 响应式高度
        result_frame = QFrame()
        result_frame.setMinimumHeight(60)  # 减少识别结果区域高度
        result_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        result_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 255, 0, 0.1);
                border-radius: 10px;
                border: 2px solid rgba(0, 255, 0, 0.3);
            }
        """)
        
        result_layout = QVBoxLayout(result_frame)
        result_layout.setContentsMargins(10, 5, 10, 5)  # 减少内边距
        result_layout.setSpacing(3)  # 减少间距
        
        result_title = QLabel("✨ 识别结果")
        result_title.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        result_title.setAlignment(Qt.AlignCenter)
        result_title.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        result_layout.addWidget(result_title)
        
        self.result_label = QLabel("等待识别...")
        self.result_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                background: transparent;
                border: none;
            }
        """)
        self.result_label.setAlignment(Qt.AlignCenter)
        self.result_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        result_layout.addWidget(self.result_label)
        
        camera_layout.addWidget(result_frame)
        
        parent_layout.addWidget(camera_frame, stretch=3)  # 调整摄像头区域比例
        
    def create_control_panel(self, parent_layout):
        """创建右侧控制面板"""
        control_frame = QFrame()
        # 重新设计的控制面板 - 适应丰富内容
        control_frame.setMinimumWidth(280)  # 增加宽度以适应新内容
        control_frame.setMaximumWidth(380)  # 增加最大宽度
        control_frame.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Expanding)
        control_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 20px;
                border: 2px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        control_layout = QVBoxLayout(control_frame)
        control_layout.setSpacing(10)  # 减少间距
        control_layout.setContentsMargins(15, 15, 15, 15)  # 减少边距
        
        # 控制面板标题
        control_title = QLabel("🎛️ 控制面板")
        control_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 16px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 5px;
            }
        """)
        control_title.setAlignment(Qt.AlignCenter)
        control_layout.addWidget(control_title)
        
        # 系统状态指示区域
        status_info_frame = QFrame()
        status_info_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.3);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        
        status_info_layout = QVBoxLayout(status_info_frame)
        status_info_layout.setSpacing(8)
        status_info_layout.setContentsMargins(15, 12, 15, 12)
        
        # 摄像头状态
        self.camera_status_label = QLabel("📷 摄像头状态")
        self.camera_status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        status_info_layout.addWidget(self.camera_status_label)
        
        self.camera_status_value = QLabel("🔴 未连接")
        self.camera_status_value.setStyleSheet("""
            QLabel {
                color: #ff6b6b;
                font-size: 12px;
                background: transparent;
                border: none;
                padding-left: 10px;
            }
        """)
        status_info_layout.addWidget(self.camera_status_value)
        
        # 识别状态
        self.recognition_status_label = QLabel("🧠 识别状态")
        self.recognition_status_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
                margin-top: 8px;
            }
        """)
        status_info_layout.addWidget(self.recognition_status_label)
        
        self.recognition_status_value = QLabel("⏸️ 待机中")
        self.recognition_status_value.setStyleSheet("""
            QLabel {
                color: #ffd93d;
                font-size: 12px;
                background: transparent;
                border: none;
                padding-left: 10px;
            }
        """)
        status_info_layout.addWidget(self.recognition_status_value)
        
        control_layout.addWidget(status_info_frame)
        
        # 主要控制按钮区域
        button_frame = QFrame()
        button_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.05);
                border-radius: 12px;
                border: 1px solid rgba(255, 255, 255, 0.1);
            }
        """)
        
        button_layout = QVBoxLayout(button_frame)
        button_layout.setSpacing(12)
        button_layout.setContentsMargins(15, 15, 15, 15)
        
        button_title = QLabel("🎮 操作控制")
        button_title.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        button_title.setAlignment(Qt.AlignCenter)
        button_layout.addWidget(button_title)
        
        # 主要控制按钮
        self.start_button = self.create_button("🚀 开始识别", "#00ff00", "#00cc00")
        self.stop_button = self.create_button("⏹️ 停止识别", "#ff4444", "#cc0000")
        
        # 连接按钮事件
        self.start_button.clicked.connect(self.start_recognition)
        self.stop_button.clicked.connect(self.stop_recognition)
        
        # 初始状态设置
        self.stop_button.setEnabled(False)
        
        button_layout.addWidget(self.start_button)
        button_layout.addWidget(self.stop_button)
        
        control_layout.addWidget(button_frame)
        
        # 快捷信息区域
        info_frame = QFrame()
        info_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 100, 150, 0.2);
                border-radius: 12px;
                border: 1px solid rgba(0, 150, 255, 0.3);
            }
        """)
        
        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(8)
        info_layout.setContentsMargins(15, 12, 15, 12)
        
        info_title = QLabel("💡 使用提示")
        info_title.setStyleSheet("""
            QLabel {
                color: #4fc3f7;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
            }
        """)
        info_title.setAlignment(Qt.AlignCenter)
        info_layout.addWidget(info_title)
        
        tips = [
            "• 确保摄像头正常连接",
            "• 保持良好的光线条件",
            "• 面部正对摄像头"
        ]
        
        for tip in tips:
            tip_label = QLabel(tip)
            tip_label.setStyleSheet("""
                QLabel {
                    color: #b3e5fc;
                    font-size: 11px;
                    background: transparent;
                    border: none;
                    padding: 2px;
                }
            """)
            info_layout.addWidget(tip_label)
        
        control_layout.addWidget(info_frame)
        
        # 添加弹性空间
        control_layout.addStretch()
        
        parent_layout.addWidget(control_frame, stretch=2)  # 增加控制面板比例
        
    def create_button(self, text, color, hover_color):
        """创建样式化按钮"""
        button = QPushButton(text)
        # 优化按钮高度以适应新布局
        button.setMinimumHeight(40)  # 适中的按钮高度
        button.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Preferred)
        button.setStyleSheet(f"""
            QPushButton {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {color}, stop:1 {hover_color});
                border: none;
                border-radius: 18px;
                color: white;
                font-size: 14px;
                font-weight: bold;
                padding: 10px 15px;
                margin: 4px;
            }}
            QPushButton:hover {{
                background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                    stop:0 {hover_color}, stop:1 {color});
            }}
            QPushButton:pressed {{
                background: {hover_color};
                padding: 8px 13px;
            }}
        """)
        return button
        
    def create_status_bar(self, parent_layout):
        """创建底部状态栏"""
        status_frame = QFrame()
        # 针对小窗口优化状态栏高度
        status_frame.setMinimumHeight(40)  # 减少状态栏高度
        status_frame.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Minimum)
        status_frame.setStyleSheet("""
            QFrame {
                background: rgba(0, 0, 0, 0.7);
                border-radius: 15px;
                border: 1px solid rgba(255, 255, 255, 0.2);
            }
        """)
        
        status_layout = QHBoxLayout(status_frame)
        status_layout.setContentsMargins(15, 8, 15, 8)  # 减少内边距
        status_layout.setSpacing(15)  # 减少间距
        
        # 系统状态
        self.system_status = QLabel("🟢 系统就绪")
        self.system_status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.system_status.setStyleSheet("""
            QLabel {
                color: #00ff00;
                font-size: 14px;
                font-weight: bold;
                background: transparent;
                border: none;
                padding: 3px;
            }
        """)
        status_layout.addWidget(self.system_status)
        
        status_layout.addStretch()
        
        # 连接状态
        self.connection_status = QLabel("📡 摄像头: 未连接")
        self.connection_status.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.connection_status.setStyleSheet("""
            QLabel {
                color: #ffaa00;
                font-size: 12px;
                background: transparent;
                border: none;
                padding: 3px;
            }
        """)
        status_layout.addWidget(self.connection_status)
        
        status_layout.addStretch()
        
        # 时间显示
        self.time_label = QLabel("⏰ 00:00:00")
        self.time_label.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Preferred)
        self.time_label.setStyleSheet("""
            QLabel {
                color: #ffffff;
                font-size: 12px;
                background: transparent;
                border: none;
                padding: 3px;
            }
        """)
        status_layout.addWidget(self.time_label)
        
        parent_layout.addWidget(status_frame)
        
    def setup_animations(self):
        """设置动画效果"""
        # 这里可以添加一些动画效果
        pass
        
    def start_recognition(self):
        """开始人脸识别"""
        try:
            # 更新UI状态
            self.start_button.setEnabled(False)
            self.stop_button.setEnabled(True)
            
            # 更新状态指示器
            self.camera_status_value.setText("🟡 连接中...")
            self.camera_status_value.setStyleSheet("""
                QLabel {
                    color: #ffd93d;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            self.recognition_status_value.setText("🔄 启动中...")
            self.recognition_status_value.setStyleSheet("""
                QLabel {
                    color: #4fc3f7;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            # 更新识别结果显示
            self.result_label.setText("正在启动识别系统...")
            
            # 启动识别线程
            self.recognition_thread.start_recognition()
            
        except Exception as e:
            self.handle_error(f"启动识别失败: {str(e)}")
            
    def stop_recognition(self):
        """停止人脸识别"""
        try:
            # 停止识别线程
            self.recognition_thread.stop_recognition()
            
            # 更新UI状态
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            # 重置状态指示器
            self.camera_status_value.setText("🔴 未连接")
            self.camera_status_value.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            self.recognition_status_value.setText("⏸️ 待机中")
            self.recognition_status_value.setStyleSheet("""
                QLabel {
                    color: #ffd93d;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            # 重置摄像头显示
            self.camera_display.clear()
            self.camera_display.setText("📷\n\n摄像头已断开\n\n点击'开始识别'按钮重新启动")
            
            # 重置识别结果
            self.result_label.setText("识别已停止")
            
        except Exception as e:
            self.handle_error(f"停止识别失败: {str(e)}")
            
    def update_camera_display(self, frame):
        """更新摄像头显示"""
        try:
            # 转换OpenCV图像为Qt格式
            height, width, channel = frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888).rgbSwapped()
            
            # 缩放图像以适应显示区域
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.camera_display.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            
            # 更新显示
            self.camera_display.setPixmap(scaled_pixmap)
            
        except Exception as e:
            print(f"更新摄像头显示失败: {str(e)}")
            
    def update_recognition_result(self, name, confidence):
        """更新识别结果"""
        try:
            if name == "Unknown":
                self.result_label.setText(f"未知人员 (置信度: {confidence:.1f})")
                self.result_label.setStyleSheet("""
                    QLabel {
                        color: #ff6b6b;
                        font-size: 14px;
                        background: transparent;
                        border: none;
                    }
                """)
            else:
                self.result_label.setText(f"识别成功: {name}\n置信度: {confidence:.1f}")
                self.result_label.setStyleSheet("""
                    QLabel {
                        color: #4caf50;
                        font-size: 14px;
                        background: transparent;
                        border: none;
                    }
                """)
        except Exception as e:
            print(f"更新识别结果失败: {str(e)}")
            
    def update_status(self, message):
        """更新系统状态"""
        try:
            if "摄像头已连接" in message:
                self.camera_status_value.setText("🟢 已连接")
                self.camera_status_value.setStyleSheet("""
                    QLabel {
                        color: #4caf50;
                        font-size: 12px;
                        background: transparent;
                        border: none;
                        padding-left: 10px;
                    }
                """)
                
            if "开始识别" in message:
                self.recognition_status_value.setText("🔄 识别中")
                self.recognition_status_value.setStyleSheet("""
                    QLabel {
                        color: #4caf50;
                        font-size: 12px;
                        background: transparent;
                        border: none;
                        padding-left: 10px;
                    }
                """)
                
        except Exception as e:
            print(f"更新状态失败: {str(e)}")
            
    def handle_error(self, error_message):
        """处理错误信息"""
        try:
            # 显示错误对话框
            QMessageBox.critical(self, "错误", error_message)
            
            # 重置UI状态
            self.start_button.setEnabled(True)
            self.stop_button.setEnabled(False)
            
            # 更新状态指示器
            self.camera_status_value.setText("🔴 连接失败")
            self.camera_status_value.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            self.recognition_status_value.setText("❌ 错误")
            self.recognition_status_value.setStyleSheet("""
                QLabel {
                    color: #ff6b6b;
                    font-size: 12px;
                    background: transparent;
                    border: none;
                    padding-left: 10px;
                }
            """)
            
            # 更新识别结果
            self.result_label.setText("识别系统出现错误")
            
        except Exception as e:
            print(f"处理错误失败: {str(e)}")
            
    def closeEvent(self, event):
        """窗口关闭事件"""
        try:
            # 确保线程正确停止
            if self.recognition_thread.isRunning():
                self.recognition_thread.stop_recognition()
            event.accept()
        except Exception as e:
            print(f"关闭窗口失败: {str(e)}")
            event.accept()

# 主函数
if __name__ == '__main__':
    app = QApplication(sys.argv)
    
    # 设置应用程序样式
    app.setStyle('Fusion')
    
    window = FaceRecognitionWidget()
    window.show()
    
    sys.exit(app.exec_())
