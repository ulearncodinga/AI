import sys
from PyQt5.QtWidgets import QApplication, QWidget, QLabel
from PyQt5.QtCore import QTimer, QDateTime

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()

    def ui_init(self):
        self.setWindowTitle("PyQt5定时器demo1")
        self.setGeometry(700, 400, 400, 150)

        # 使用label获取当前时间
        self.showtime_label = QLabel("PyQt5定时器demo1", self)
        self.showtime_label.setGeometry(100, 50, 130, 30)

        # 引入定时器
        self.timer = QTimer(self)
        # 设置超时时间  1S  参数单位是毫秒
        self.timer.setInterval(1000)
        # timeout 定时器产生的信号
        self.timer.timeout.connect(self.update_time)
        # 定时器需要启动，否则不工作
        self.timer.start()

    def update_time(self):
        """
        获取当前时间
        """
        # 时间格式：yyyy   4位年份 ， MM 2位月份， dd  2位天， hh 24小时制，mm 分钟，ss 秒，zzz 毫秒
        self.current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.showtime_label.setText(self.current_time)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec_())