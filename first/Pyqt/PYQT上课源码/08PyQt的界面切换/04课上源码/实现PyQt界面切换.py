"""
    PyQt界面切换
"""

import sys
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication, QWidget, QPushButton, QMessageBox

class indexWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()
        self.slot_init()

    def ui_init(self):
        self.setWindowTitle("主界面")

        self.login_btn = QPushButton("登录", self)
        self.login_btn.setGeometry(200, 50, 100, 40)

        self.reg_btn = QPushButton("注册", self)
        self.reg_btn.setGeometry(200, 120, 100, 40)

        self.exit_btn = QPushButton("退出", self)
        self.exit_btn.setGeometry(200, 190, 100, 40)

        # 创建登录界面的对象
        self.login_show = loginWidget()
        # 创建注册界面的对象
        self.reg_show = regWidget()

    def slot_init(self):
        self.login_btn.clicked.connect(self.login_wid)
        self.reg_btn.clicked.connect(self.reg_wid)
        self.exit_btn.clicked.connect(self.exit_close)

        # 登录界面的信号关联
        self.login_show.send_index.connect(self.show)

        # 注册界面的信号关联
        self.reg_show.send_index.connect(self.show)

        # 传参
        self.reg_show.send_arg_login.connect(self.login_show.recv_meg)


    def login_wid(self):
        self.close()
        # 显示登录界面
        self.login_show.show()

    def reg_wid(self):
        self.close()
        # 显示注册界面
        self.reg_show.show()

    def exit_close(self):
        self.result = QMessageBox.question(self, "提示", "确定要退出吗？", QMessageBox.Yes | QMessageBox.No)
        if self.result == QMessageBox.Yes:
            # 0表示正常退出 ， 1表示一般问题（不影响系统运行）
            sys.exit(0)
        else:
            pass


class loginWidget(QWidget):
    # 自定义信号
    send_index = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录界面")
        # 设置返回按钮
        self.exit_index_btn = QPushButton("返回主界面", self)
        self.exit_index_btn.setGeometry(200, 180, 100, 40)
        self.exit_index_btn.clicked.connect(self.indexwid_show)

    def indexwid_show(self):
        self.close()
        # 显示主界面
        self.send_index.emit()

    def recv_meg(self, arg):
        print(f'接收到的参数：{arg}')
        print(f'字典的值：{arg["这是来自于注册界面的参数"]}')


class regWidget(QWidget):
    # 自定义信号
    send_index = pyqtSignal()
    send_arg_login = pyqtSignal(dict)

    def __init__(self):
        super().__init__()
        self.setWindowTitle("注册界面")
        # 设置返回按钮
        self.exit_index_btn = QPushButton("返回主界面", self)
        self.exit_index_btn.setGeometry(200, 180, 100, 40)
        self.exit_index_btn.clicked.connect(self.indexwid_show)

    def indexwid_show(self):
        self.close()
        # 显示主界面
        self.send_index.emit()
        # 点击退出界面的时候，触发信号，发送参数
        self.send_arg_login.emit({"这是来自于注册界面的参数": 123})



if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = indexWidget()
    window.show()
    sys.exit(app.exec_())