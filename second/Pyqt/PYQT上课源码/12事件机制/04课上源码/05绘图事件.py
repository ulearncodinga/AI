"""
    绘图事件
"""
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtGui import QPainter, QPixmap, QPen
from PyQt5.QtWidgets import QWidget, QApplication



class paintWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("绘图事件")
        self.x = 200
        self.y = 200
        self.r = 100
        # 鼠标追踪
        self.setMouseTracking(True)

    def paintEvent(self, a0):
        """
        重写绘图事件
        :param a0:
        :return:
        """
        # 创建一个画家对象，
        painter = QPainter(self)
        # 实现背景图的自适应
        painter.drawPixmap(0, 0, self.width(), self.height(), QPixmap("../images/background.png").scaled(self.size()))

        # 绘制圆
        # 准备画笔
        pen = QPen()
        # 设置画笔颜色
        pen.setColor(Qt.red)
        # 设置画笔的粗细
        pen.setWidth(4)
        # 将画笔设置给画家
        painter.setPen(pen)
        # 参数内容，起点的x , y 终点的x, y
        painter.drawLine(100, 100, 300, 300)
        # 绘制椭圆
        painter.drawEllipse(self.x, self.y, self.r, self.r)

    def mouseMoveEvent(self, a0):
        self.x = a0.x()
        self.y = a0.y()
        self.update()

    def keyPressEvent(self, a0):
        print("键盘输入事件")
        if a0.key() == Qt.Key_Up:
            self.y -= 5
        elif a0.key() == Qt.Key_Down:
            self.y += 5
        elif a0.key() == Qt.Key_Left:
            self.x -= 5
        elif a0.key() == Qt.Key_Right:
            self.x += 5
        self.update()

    def wheelEvent(self, a0):
        print("滑轮事件")
        # 判断滚轮的方向
        if a0.angleDelta().y() > 0:
            print("向上，远离用户的方向")
            # 放大圆
            self.r += 5
        else:
            print("向下")
            self.r -= 5
        self.update()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = paintWidget()
    w.show()
    sys.exit(app.exec_())