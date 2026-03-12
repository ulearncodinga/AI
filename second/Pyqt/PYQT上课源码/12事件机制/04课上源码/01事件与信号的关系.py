"""
    事件与信号的关系
"""
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QPushButton

class myButton(QPushButton):
    def __init__(self):
        super().__init__()

    def mousePressEvent(self, e):
        """
        重写鼠标点击处理函数
        :param e: 鼠标点击事件的对象
        :return:
        """
        print('执行事件处理函数')
        # 触发按钮信号
        self.clicked.emit()


class Widget(QWidget):
    def __init__(self):
        super().__init__()

        self.setWindowTitle("事件与信号的关系")
        self.resize(350, 200)

        # self.btn = QPushButton("点击", self)
        self.btn = myButton()
        self.btn.setParent(self)
        self.btn.setText("点击")
        self.btn.move(130, 80)

        self.btn.clicked.connect(self.btn_slot)

    def btn_slot(self):
        print("执行按钮槽函数")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())


