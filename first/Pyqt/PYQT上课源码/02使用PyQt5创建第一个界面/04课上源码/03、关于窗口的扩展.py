"""
生成的界面居中，且位于当前鼠标活动的显示器上
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget
from PyQt5.QtGui import QCursor

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("居中的应用程序")

    # 监测当前鼠标光标在那个显示器上
    cursor_pos = QCursor.pos()
    # 把鼠标的位置传入app, screenAt方法作用：返回当前光标所在的屏幕的对象
    current_screen = app.screenAt(cursor_pos)

    if current_screen is not None:
        # 获取当前屏幕的几何信息
        screen_rect = current_screen.availableGeometry()
        print(f'当前屏幕大小：{screen_rect}')
        # 设置创建的窗口的大小为屏幕的50%，使用整数
        width = int(screen_rect.width() * 0.5)
        high = int(screen_rect.height() * 0.5)
        # 设置为窗口的大小
        window.resize(width, high)

        # 计算居中的位置
        # screen_rect.left()当前屏幕可用几何形状的左边界坐标
        x = int(screen_rect.width() / 2 - width / 2 + screen_rect.left())
        y = int(screen_rect.height() / 2 - high / 2 + screen_rect.top())
        # 设置窗口位置到坐标（x, y）
        window.move(x, y)

    window.show()
    app.exec()
