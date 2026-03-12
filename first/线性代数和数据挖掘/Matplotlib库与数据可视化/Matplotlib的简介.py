# Matplotlib 绘图库
# Matplotlib 库的安装
'''
pip install matplotlib
'''
import sys
import random
from PyQt5.QtWidgets import (QApplication, QMainWindow, QWidget, QVBoxLayout,
                             QProgressBar, QPushButton, QMessageBox, QLabel)
from PyQt5.QtCore import QTimer, Qt


class DownloadSimulator(QMainWindow):
    def __init__(self):
        super().__init__()
        self.initUI()
        self.download_progress = 0
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_progress)

    def initUI(self):
        """初始化用户界面"""
        self.setWindowTitle('资源下载模拟器')
        self.setGeometry(300, 300, 400, 200)

        # 创建中央部件和布局
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 创建标题标签
        title_label = QLabel('资源下载进度模拟')
        title_label.setAlignment(Qt.AlignCenter)
        title_label.setStyleSheet("font-size: 16px; font-weight: bold; margin: 10px;")
        layout.addWidget(title_label)

        # 创建进度条
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)  # 设置范围0-100
        self.progress_bar.setValue(0)  # 初始值为0
        self.progress_bar.setFormat('当前进度: %p%')  # 显示百分比
        self.progress_bar.setStyleSheet(self.get_progressbar_style())
        layout.addWidget(self.progress_bar)

        # 创建状态标签
        self.status_label = QLabel('准备下载...')
        self.status_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.status_label)

        # 创建按钮
        self.start_button = QPushButton('开始下载')
        self.start_button.clicked.connect(self.start_download)
        layout.addWidget(self.start_button)

        # 创建重置按钮
        self.reset_button = QPushButton('重置')
        self.reset_button.clicked.connect(self.reset_download)
        self.reset_button.setEnabled(False)
        layout.addWidget(self.reset_button)

    def get_progressbar_style(self):
        """返回进度条的样式表"""
        return """
            QProgressBar {
                border: 2px solid #2D2D30;
                border-radius: 5px;
                text-align: center;
                color: white;
                font-weight: bold;
                background-color: #1E1E1E;
                height: 25px;
            }
            QProgressBar::chunk {
                background-color: QLinearGradient(
                    x1: 0, y1: 0, 
                    x2: 1, y2: 0, 
                    stop: 0 #FF5F6D, 
                    stop: 1 #FFC371
                );
                border-radius: 3px;
            }
        """

    def start_download(self):
        """开始下载模拟"""
        if self.download_progress >= 100:
            QMessageBox.information(self, '提示', '下载已完成，请重置后重新开始!')
            return

        self.start_button.setEnabled(False)
        self.reset_button.setEnabled(False)
        self.status_label.setText('下载中...')
        self.timer.start(1000)  # 每秒更新一次

    def update_progress(self):
        """更新下载进度"""
        # 随机生成增量（0-10之间）
        increment = random.randint(0, 10)
        self.download_progress += increment

        # 确保不超过100%
        if self.download_progress > 100:
            self.download_progress = 100

        # 更新进度条
        self.progress_bar.setValue(self.download_progress)

        # 更新状态标签
        self.status_label.setText(f'已下载: {self.download_progress}%')

        # 检查是否完成
        if self.download_progress >= 100:
            self.timer.stop()
            self.status_label.setText('下载完成!')
            self.reset_button.setEnabled(True)
            QMessageBox.information(self, '完成', '资源下载完成!')

    def reset_download(self):
        """重置下载进度"""
        self.download_progress = 0
        self.progress_bar.setValue(0)
        self.status_label.setText('准备下载...')
        self.start_button.setEnabled(True)
        self.reset_button.setEnabled(False)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = DownloadSimulator()
    window.show()
    sys.exit(app.exec_())