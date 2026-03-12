import sys

from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLCDNumber, QMessageBox
from PyQt5.QtCore import QTimer, QTime

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()

    def ui_init(self):
        # 倒计时的计数值
        self.count = 5
        # 设置时间格式QTime,不支持毫秒的，仅支持小时：分钟：秒
        self.time = QTime(0, 0, 0)
        # 将计数值转换为时间格式
        self.start_time = self.time.addSecs(self.count).toString("hh:mm:ss")

        self.show_lcd = QLCDNumber(self)
        # 设置显示位数
        self.show_lcd.setDigitCount(8)
        # 设置初始值
        self.show_lcd.display(self.start_time)

        # 设置LCD的位置和大小
        self.show_lcd.setGeometry(100, 30, 200, 30)

        # 按钮
        self.btn = QPushButton("开始倒计时", self)
        self.btn.setGeometry(120, 80, 160, 30)
        # 按钮被点击，打开定时器
        self.btn.clicked.connect(self.start_timer)

        # 引入定时器
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.update_time)

    def start_timer(self):
        """
        一个按钮完成两个工作
        :return:
        """
        if self.btn.text() == "开始倒计时":
            self.timer.start()
            self.btn.setText("暂停倒计时")

        elif self.btn.text() == "暂停倒计时":
            self.timer.stop()
            self.btn.setText("开始倒计时")

    def update_time(self):
        self.count -= 1
        if self.count < 0:
            QMessageBox.information(self, "倒计时", "比赛结束！")
            self.timer.stop()
            self.btn.setText("开始倒计时")
        else:
            self.current_time = self.time.addSecs(self.count).toString("hh:mm:ss")
            self.show_lcd.display(self.current_time)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    sys.exit(app.exec_())