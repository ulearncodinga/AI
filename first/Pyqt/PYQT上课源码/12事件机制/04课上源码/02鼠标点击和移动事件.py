"""
    鼠标点击和移动事件
"""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPixmap
from PyQt5.QtWidgets import QWidget, QApplication, QLabel

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("鼠标点击和移动事件")
        self.lab = QLabel("点击鼠标", self)
        self.lab.move(100, 70)

        self.pic_lab = QLabel(self)
        self.pic_lab.setGeometry(100, 120, 200, 200)

        self.pic_lab.setPixmap(QPixmap("../images/light_on.jpg").scaled(self.pic_lab.size()))

        # 打开鼠标追踪功能
        self.setMouseTracking(True)


    def mousePressEvent(self, a0):
        """
        鼠标点击事件处理函数
        :param a0: 是 QMouseEvent 的对象
        :return:
        """
        # 获取鼠标位置
        # print(a0.pos())

        x = a0.x()
        # print(x)
        y = a0.y()
        # print(y)
        self.lab.move(x, y)

        if a0.button() == Qt.LeftButton:
            print("点击的是左键")
        elif a0.button() == Qt.RightButton:
            print("点击的是右键")
        elif a0.button() == Qt.MidButton:
            print("点击的是中间按键")

    def mouseMoveEvent(self, a0):
        """
        鼠标移动事件处理函数
        :param a0: 鼠标移动事件的对象
        :return:
        """
        print("鼠标移动了")
        x = a0.x()
        y = a0.y()
        self.pic_lab.setGeometry(x, y, 200, 200)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())