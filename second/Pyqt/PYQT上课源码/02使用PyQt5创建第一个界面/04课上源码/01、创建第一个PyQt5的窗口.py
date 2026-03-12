"""
使用PyQt5创建的第一个窗口
"""

import sys
# 导入界面控件设计模块
from PyQt5.QtWidgets import QApplication, QWidget

if __name__ == '__main__':
    # 创建QApplication的实例
    # sys.argv，本质是一个list，包括了命令行参数，sys.argv[0],是被执行的脚本的名称，后续的元素则是传递给脚本的参数。
    app = QApplication(sys.argv)

    # 创建窗口对象
    window = QWidget()

    # 窗口标题
    window.setWindowTitle("第一个PyQt5的应用程序")

    # # 设置窗口的尺寸 宽，  高
    # window.resize(400, 300)
    #
    # # 设置窗口移动
    # window.move(100, 100)

    # 显示窗口
    window.show()

    # 监测整个程序所收到的用户的交互信息，直到调用sys.exit()退出
    app.exec()

