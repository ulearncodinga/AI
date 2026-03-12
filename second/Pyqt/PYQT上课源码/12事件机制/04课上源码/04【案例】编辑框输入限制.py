"""
    实现编辑框输入限制
"""
import sys

from PyQt5.QtCore import QEvent
from PyQt5.QtGui import QKeyEvent
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit

class myLineEdit(QLineEdit):
    def __init__(self):
        super().__init__()

    def keyPressEvent(self, a0):
        """
        通过重写键盘输入事件处理函数，从而达到限制输入的目的
        汉字拼音并不会触发 keyPressEvent
        无法达到限制汉字拼音的输入
        :param a0:
        :return:
        """
        # pass
        data = a0.text()
        print(f"按下的键：{data}")
        if "0" <= data <= "9":
            self.setText(self.text() + data)

    # def event(self, a0):
    #     """
    #     重写事件分发函数
    #     :param a0:
    #     :return:
    #     """
    #     if a0.type() == QEvent.KeyPress:
    #         print("这是键盘输入事件")
    #         # 如果想进行数字输入，那么解开下行注释
    #         # self.keyPressEvent(QKeyEvent(a0))
    #         return True
    #     else:
    #         return super(myLineEdit, self).event(a0)


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("编辑框的输入限制")

        self.edit = myLineEdit()
        self.edit.setParent(self)
        self.edit.setGeometry(100, 50, 200, 40)

        # 给控件安装事件过滤器
        self.edit.installEventFilter(self)

    def eventFilter(self, a0, a1):
        """
        重写事件过滤函数
        :param a0: 控件对象
        :param a1: 事件
        :return:
        """
        if a0 == self.edit:
            if a1.type() == QEvent.KeyPress:
                print('事件过滤函数中的，输入事件')
                return True
            else:
                return super(Widget, self).eventFilter(a0, a1)
        else:
            return super(Widget, self).eventFilter(a0, a1)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())