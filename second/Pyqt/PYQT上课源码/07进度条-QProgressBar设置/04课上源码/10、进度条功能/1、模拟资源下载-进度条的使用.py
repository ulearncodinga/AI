"""
    模拟资源下载
"""
import random
import sys
from PyQt5.QtCore import QTimer
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox, QProgressBar

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()
        self.slot_init()

    def ui_init(self):
        self.setWindowTitle("模拟资源下载")
        # 设置进度条控件
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setGeometry(100, 30, 300, 30)
        # 设置进度条的取值范围
        self.progress_bar.setRange(0, 100)
        # 给进度条设置初始值
        self.progress_bar.setValue(0)

        self.btn = QPushButton("开始下载", self)
        self.btn.setGeometry(150, 80, 150, 30)

    def slot_init(self):
        # 按钮功能是开启关闭定时器
        self.btn.clicked.connect(self.start_timer)

        # 设置定时器
        self.timer = QTimer(self)
        # 设置定时器的超时事件，1S
        self.timer.setInterval(1000)
        # 定时器关联功能
        self.timer.timeout.connect(self.update_value)

    def start_timer(self):
        if self.btn.text() == "开始下载":
            # 开始下载，打开定时器
            self.timer.start()
            self.btn.setText("暂停下载")
            print("开始下载，定时器开启")
        else:
            self.timer.stop()
            self.btn.setText("开始下载")
            print("暂停下载，定时器关闭")


    def update_value(self):
        # 模拟资源下载的进度
        self.random_num = random.randint(0, 10)
        print(f'随机生成的值：{self.random_num}')
        # 获取当前的进度值
        self.cur_num = self.progress_bar.value()
        print(f'当前进度条的值：{self.cur_num}')
        # 更新的值
        self.update_num = self.cur_num + self.random_num
        print(f'更新的值：{self.update_num}')

        if self.cur_num + self.random_num >= 100:
            # 表示下载完成
            self.progress_bar.setValue(100)
            # 停止定时器
            self.timer.stop()
            QMessageBox.information(self, "资源下载", "资源下载完成！")
        else:
            # 将更新的值，放入进度条内
            self.progress_bar.setValue(self.update_num)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    sys.exit(app.exec_())
