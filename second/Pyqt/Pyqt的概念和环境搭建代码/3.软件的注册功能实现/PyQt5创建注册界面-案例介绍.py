'''
需求：创建一个界面，用于用户注册信息，包括用户名与密码，并保存用户信息。
要求：用户名与密码不能为空，用户名使用数字注册且长度为11位，密码长度为6位）,清除功能
需要的组件： 一个按钮 QPushButton，两个标签 QLabel，两个编辑框QLineEdit，对话框
QMessageBox
实现方式：手写代码完成开发
'''
import sys
from PyQt5 import QtGui,QtCore
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QLineEdit,QPushButton,QMessageBox

class MyWindow(QWidget):
    '''
    注册界面,及实现功能
    '''
    def __init__(self):
        super().__init__()
        self.setWindowTitle("注册界面")
        #初始化UI
        self.ui_init()
        #初始化槽函数
        self.slot_init()

    def ui_init(self):
        # 账号/用户名  QLabel
        self.username = QLabel("用户名",self)
        self.username.setGeometry(10,20,100,40)

        #用户名输入框
        self.username_edit = QLineEdit(self)
        self.username_edit.setGeometry(150,20,200,40)


        #setmaxlength,限制输入的最大长度,直截取前十一位
        self.username_edit.setMaxLength(11)


        #限制用户名输入框只能输入数字
        #验证器,三种,整数验证器,正则表达式验证器,浮点数验证器
        int_validator = QtGui.QIntValidator()#整数
        # int_validator = Qtgui.QRegExpValidator()#正则表达式验证器
        # int_validator = QtGui.QDoubleValidator()#浮点数验证器
        #正则表达式
        #^表示匹配必须从字符串的开始处开始
        #\d 表示可以匹配任何数字字符{0,9}
        # {0,11}表示每个数字可以出现最大次数是11次,最小为0次
        #$ 表示必须到字符串结束的时候结束
        reg_exp =  QtCore.QRegExp("^\d\w{0,11}$")
        reg_exp_validator = QtGui.QRegExpValidator(reg_exp,self.username_edit)


        self.username_edit.setValidator(int_validator)
        self.username_edit.setValidator(reg_exp_validator)




        # 密码
        self.password = QLabel("密码", self)
        self.password.setGeometry(10, 80, 100, 40)
        # 密码输入框
        self.password_edit = QLineEdit(self)
        self.password_edit.setGeometry(150, 80, 200, 40)



        #限制输入长度
        self.password_edit.setMaxLength(6)


        #设置密码显示模式#这个模式为密码模式,把密码显示为黑点,NoEcho可以输入但是不显示,Normal 什么也不改,默认模式
        self.password_edit.setEchoMode(QLineEdit.Password)









        #注册按钮
        self.reg_btn = QPushButton("注册",self)
        self.reg_btn.setGeometry(10,140,160,40)
        #清空按钮
        self.clear_btn = QPushButton("清空",self)
        self.clear_btn.setGeometry(190,140,160,40)

    def slot_init(self):
        self.reg_btn.clicked.connect(self.reg_slot)
        self.clear_btn.clicked.connect(self.clear_slot)

    def reg_slot(self):
        """
        注册按钮关联的槽函数
        :return:
        """
        username = self.username_edit.text()
        password = self.password_edit.text()
        if username == ""or password == "":
            #消息提醒框
            QMessageBox.warning(self,"注册","用户名或者密码不能为空")
        elif 0< len(username) <11:
            QMessageBox.warning(self,"注册","用户名输入长度不符合要求")
        elif 0< len(password) < 6:
            QMessageBox.warning(self,"注册","密码输入长度不符合要求")
        else:
            #将数据写入文件中
            with open("user.txt",mode="a+",encoding="utf-8") as f:
                info = username + "," + password + "\n"
                f.write(info)
                QMessageBox.warning(self,"注册","恭喜注册完成!")


    def clear_slot(self):
        self.username_edit.clear()
        self.password_edit.clear()





if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    #当窗口关闭时,事件循环跟着结束,程序自然退出.(简单程序可以使用)
    # app.exec()
    #建议使用这一种  可以确保程序释放所有资源
    sys.exit(app.exec_())