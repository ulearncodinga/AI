"""
    定时器事件
"""
import sys
from PyQt5.QtCore import QDateTime
from PyQt5.QtWidgets import QWidget, QApplication

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("定时器事件")

        # 启动定时器
        self.startTimer(1000)

    def timerEvent(self, a0):
        print("now_time = {}".format(QDateTime.currentDateTime().toString("hh:mm:ss")))



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())