"""
    样式表与调色板
"""

import sys

from PyQt5.QtGui import QPalette, QBrush, QPixmap
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QLabel

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("背景图片设置")

        self.btn = QPushButton("按钮", self)
        self.btn.setGeometry(100, 20, 200, 50)
        self.btn.setStyleSheet("border-image:url(../images/btn.png)")

        # 调色板添加界面背景图片
        self.palette = QPalette()
        # QPalette.Background是一个颜色角色，表示窗口背景色
        # QBrush接收一个QPixmap对象作为图像
        # .scaled(self.size()作用，将图片缩小到widget的大小
        self.palette.setBrush(QPalette.Background, QBrush(QPixmap("../images/HQYJ.png").scaled(self.size())))
        self.setPalette(self.palette)

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    sys.exit(app.exec_())