'''
从主界面到登录界面  从主界面到注册界面
因为此时登录界面对象和注册界面对象还不存在，先创建，该显示的时候直接调用显示即可
但是 从登录界面返回到主界面 从注册界面返回到主界面的时候，此时主界面已经存在了，没必要再重新
创建了，直接使用即可
'''
'''
登录界面和注册界面没法引入主界面。通过发送信号的方式来显示主界面
要想实现界面之间的通信，当下使用 PYQT的核心机制：信号与槽是最合适的选择
'''

#方案
"""
使用信号与槽的机制去实现界面与界面之间的通信   因为主界面、登录界面、注册界面（其实在代码中就是
类）都是自定义的。
pyqtSignal是一个用于定义信号的类，它是QtCore模块的一部分。信号是Qt中用于对象间通信的一种机
制，允许一个对象通知其他对象某些事情发生了。信号通常与槽（slot）一起使用，槽是处理信号的方法。
  自定义信号格式： 信号名 = pyqtSignal()
  信号如何触发：  信号名.emit()
  信号与槽函数的关联： 发送者.信号.connect(接收者.槽函数)
"""
import sys
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QMessageBox


class indexWidget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()
        self.slot_init()


    def ui_init(self):
        self.setWindowTitle("主界面")
        self.login_btn = QPushButton("登录",self)
        self.login_btn.setGeometry(200,50,100,40)

        self.reg_btn =QPushButton("注册",self)
        self.reg_btn.setGeometry(200,120,100,40)

        self.exit_btn = QPushButton("退出",self)
        self.exit_btn.setGeometry(200,190,100,40)

        # 创建登录界面的对象
        self.login_show = loginWidget()
        # 创建注册界面的对象
        self.reg_show = regWidget()

    def slot_init(self):
        self.login_btn.clicked.connect(self.login_wid)
        self.reg_btn.clicked.connect(self.reg_wid)
        self.exit_btn.clicked.connect(self.exit_wid)

        # 登陆界面的信号
        self.login_show.send_index.connect(self.show)

        #注册界面的信号
        self.reg_show.send_index.connect(self.show)

        #传参
        self.reg_show.send_arg_login.connect(self.login_show.recv_meg)



    def login_wid(self):
        self.close()
        #显示登录界面
        self.login_show.show()

    def reg_wid(self):
        self.close()
        #显示注册界面
        self.reg_show.show()
    def exit_wid(self):
        self.result = QMessageBox.question(self,"提示","确定要退出嘛?",QMessageBox.Yes|QMessageBox.No)
        if self.result == QMessageBox.Yes:
            #0表示正常退出,1表示一般问题(不影响程序运行)
            sys.exit(0)
        else:
            pass

class loginWidget(QWidget):
    #创建自定义的信号
    send_index = pyqtSignal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("登录界面")
        #设置返回按钮
        self.exit_index_btn = QPushButton("返回主界面",self)
        self.exit_index_btn.setGeometry(200,180,100,40)
        self.exit_index_btn.clicked.connect(self.indexwid_show)


    def indexwid_show(self):
        self.close()
        #显示主界面
        self.send_index.emit()



    def recv_meg(self,arg,arg1):
        print(f"接收到的参数:{arg}{arg1}")

class regWidget(QWidget):
    #自定义信号
    send_index = pyqtSignal()
    send_arg_login = pyqtSignal(str,int)


    def __init__(self):
        super().__init__()
        self.setWindowTitle("注册界面")
        self.exit_index_btn = QPushButton("返回主界面",self)
        self.exit_index_btn.setGeometry(200,180,100,40)
        self.exit_index_btn.clicked.connect(self.index_show)

    def index_show(self):
        self.close()
        self.send_index.emit()
        #点击退出界面的时候,触发信号,发送参数
        self.send_arg_login.emit("这是来自于注册界面的参数",123)








if __name__ == '__main__':
    app=QApplication(sys.argv)
    window = indexWidget()
    window.show()
    sys.exit(app.exec_())




















