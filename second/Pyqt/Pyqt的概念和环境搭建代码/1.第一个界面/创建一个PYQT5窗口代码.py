
# import sys
# #导入界面控件设计模块
# from PyQt5.QtWidgets import QApplication,QWidget
#
# if __name__ == '__main__':
# #创建QApplication的实例
# #sys.argv,本质是一个list,包括了命令行参数,sys.argv[0].是被执行的脚本的名称.后续的元素则是传递给脚本的参数
#     app = QApplication(sys.argv)
# #创建窗口对象
#     window = QWidget()
# #创建窗口标题
#     window.setWindowTitle("我的第一个PyQt5的应用程序")
# #设置窗口的尺寸      宽度,高度
#     window.resize(1000,1000)
# #设置窗口移动      横向  纵向
#     window.move(1000,100)
#
# #显示窗口
#     window.show()
# #检测整个程序所收到的用户的交互信息,直到调用sys.exit()退出
#     app.exec()









import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = QWidget()
    window.setWindowTitle("标题名称")
    window.resize(1000,1000)
    window.move(100,101)
    label = QLabel("你好，这是一个窗口!", window)
    label.move(150, 130)
    window.show()
    app.exec()


# import sys
# from PyQt5.QtWidgets import QApplication,QWidget
#
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = QWidget()
#     window.setWindowTitle("我要写一个标题")
#     window.resize(100,500)
#     window.move(10,20)
#     window.show()
#     app.exec()






























