import os.path
import sys
from PyQt5 import QtGui,QtCore
from PyQt5.QtGui import QIcon,QPixmap,QMovie
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtSql import QSqlQuery,QSqlDatabase
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QLineEdit,QPushButton,QMessageBox

class MySql:
    def __init__(self):
        '''
        连接数据库
        '''
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("users1.db")
            self.database.open()
        except Exception as e:
            print(f"无法连接,error:{e}")
    def operation_sql(self,sql):
        self.query = QSqlQuery()
        self.query.exec_(sql)



    def selectData(self,sql):
        '''
        查询
        :return:
        '''
        self.query = QSqlQuery()
        self.query.exec_(sql)
        self.result = self.query.record()
        while self.query.next():
            for i in range(self.result.count()):
                print(str(self.query.value(i)))
    def close_db(self):
        self.db = QSqlDatabase.database()
        self.db.close()















class MainWindow(QWidget):
    logout_signal = pyqtSignal()
    def __init__(self,username):
        super().__init__()
        self.username = username
        self.init_ui()
    def init_ui(self):
        self.setWindowTitle("购物管理系统 - 主界面")
        self.resize(800,600)

        self.product_btn = QPushButton('商品管理',self)
        self.product_btn.setGeometry(50,100,150,50)

        self.order_btn = QPushButton("订单管理", self)
        self.order_btn.setGeometry(50, 170, 150, 50)

        self.user_btn = QPushButton("用户管理", self)
        self.user_btn.setGeometry(50, 240, 150, 50)

        self.logout_btn = QPushButton("退出登录", self)
        self.logout_btn.setGeometry(50, 310, 150, 50)

        #连接信号和槽
        self.logout_btn.clicked.connect(self.logout)

    def logout(self):
        self.logout_signal.emit()
        self.close()


class MyWindow(QWidget):
    #添加一个信号,通知登陆成功

    login_success = pyqtSignal(str)
    def __init__(self):

        super().__init__()
        #存储到文件
        # self.user_file = "user.txt"

        #存储到数据库,数据库初始化
        self.init_database()

        #设置窗口大小
        self.resize(600,400)

        self.main_window = None #添加主窗口引用

        self.ui_init()
        self.slot_init()
        self.setWindowTitle("购物管理系统")
        #连接登录成功的信号
        self.login_success.connect(self.show_main_window)

        #添加的图片居中
        self.set_background_image()


        self.btn = QPushButton(self)
        self.btn.setGeometry(250,20,100,50)
        self.btn.setStyleSheet("border-image:url(../images/HQYJ.png)")

        self.btnusername = QPushButton(self)
        self.btnusername.setGeometry(150, 80, 40, 40)
        self.btnusername.setStyleSheet("border-image:url(../images/OIP-C.webp)")

        self.btnpassword = QPushButton(self)
        self.btnpassword.setGeometry(150, 140, 40, 40)
        self.btnpassword.setStyleSheet("border-image:url(../images/password.webp)")

    def init_database(self):
        #创建连接
        self.db = QSqlDatabase.addDatabase('QSQLITE')
        self.db.setDatabaseName('users1.db')#文件名
        if not self.db.open():
            QMessageBox.critical(self,'数据库错误','无法打开数据库:'+self.db.lastError().text())
            return False
        query = QSqlQuery()
        query.exec_(
            """
            CREATE TABLE IF NOT EXISTS users (
                username VARCHAR(20) UNIQUE NOT NULL,
                password VARCHAR(20) NOT NULL
            )
            """
        )
        if query.lastError().isValid():
            QMessageBox.critical(self,'数据库错误','创建表失败:' + query.lastError().text())
            return False
        return True






    def set_background_image(self):
        """设置背景图片并使其居中显示"""
        # 创建背景标签并设置为窗口大小
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, self.width(), self.height())

        # 加载背景图片
        pixmap = QPixmap("../images/background.png")  # 替换为你的背景图片路径
        if not pixmap.isNull():
            # 保持图片比例缩放并居中显示
            scaled_pixmap = pixmap.scaled(#检查图片是否存在且有效
                self.background_label.size()#缩放到标签大小
                # QtCore.Qt.KeepAspectRatio,#纵横比设置
                # QtCore.Qt.SmoothTransformation#平滑设置
            )
            self.background_label.setPixmap(scaled_pixmap)#将scaled_pixmap设置为标签background_label的内容
            self.background_label.setAlignment(QtCore.Qt.AlignCenter)#居中对齐

        # 将背景置于底层
        self.background_label.lower()

    #将整个程序的窗口居中
    def center_window(self):
        frame_geometry = self.frameGeometry()
        center_point = QApplication.desktop().availableGeometry().center()
        frame_geometry.moveCenter(center_point)
        self.move(frame_geometry.topLeft())

    def resizeEvent(self, event):
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())
            # 重新调整背景图片大小
            pixmap = self.background_label.pixmap()
            if pixmap and not pixmap.isNull():
                scaled_pixmap = pixmap.scaled(
                    self.background_label.size(),
                    QtCore.Qt.KeepAspectRatio,
                    QtCore.Qt.SmoothTransformation
                )
                self.background_label.setPixmap(scaled_pixmap)
        super().resizeEvent(event)

    def ui_init(self):
        #用户名
        self.username = QLabel('用户名',self)
        self.username.setGeometry(100,80,100,40)
        #输入框
        self.username_edit = QLineEdit(self)
        self.username_edit.setGeometry(200,80,250,40)

        #setmaxlength,限制输入的最大长度,直截取前11位
        self.username_edit.setMaxLength(8)
        reg_exp = QtCore.QRegExp("^[a-zA-Z0-9]{1,8}$")
        reg_exp_validator = QtGui.QRegExpValidator(reg_exp,self.username_edit)


        self.username_edit.setValidator(reg_exp_validator)#验证器

        #密码设置
        self.password = QLabel('密码',self)
        self.password.setGeometry(100,140,100,40)

        #输入密码框
        self.password_edit = QLineEdit(self)
        self.password_edit.setGeometry(200,140,250,40)

        self.password_edit.setMaxLength(8)
        #把密码设置成黑点
        self.password_edit.setEchoMode(QLineEdit.Password)

        #登录按钮
        self.center_btn = QPushButton("登录", self)
        self.center_btn.setGeometry(150, 220, 120, 50)
        # 注册按钮
        self.reg_btn = QPushButton("注册", self)
        self.reg_btn.setGeometry(330, 220, 120, 50)



    def slot_init(self):
        self.reg_btn.clicked.connect(self.reg_slot)
        self.center_btn.clicked.connect(self.center_slot)





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
        elif 0< len(username) < 8:
            QMessageBox.warning(self,"注册","用户名输入长度不符合要求")
        elif 0< len(password) < 8:
            QMessageBox.warning(self,"注册","密码输入长度不符合要求")
        else:
            if self.is_username_exists(username):
                QMessageBox.warning(self, "注册", "用户名已存在，请选择其他用户名")
                return
            #将数据写入数据库
            query = QSqlQuery()
            query.prepare("INSERT INTO users (username, password) VALUES (?, ?)")
            query.addBindValue(username)
            query.addBindValue(password)
            if query.exec_():
                QMessageBox.information(self, "注册", "恭喜注册完成!")
            else:
                QMessageBox.critical(self, "注册错误", "注册失败: " + query.lastError().text())

            #将数据写入文件中
            # with open("user.txt",mode="a+",encoding="utf-8") as f:
            #     info = username + "," + password + "\n"
            #     f.write(info)
            #     QMessageBox.information(self,"注册","恭喜注册完成!")

    def center_slot(self):
        # 获取输入的用户名和密码,去除空格
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()

        # 2. 简单验证：输入不能为空
        if username == "" or password == "":
            QMessageBox.warning(self, "登录失败", "用户名或密码不能为空！")
            return  # 直接结束函数，不往下执行
        query = QSqlQuery()
        query.prepare("SELECT * FROM users WHERE username = ? AND password = ?")
        query.addBindValue(username)
        query.addBindValue(password)

        if query.exec_() and query.next():
            # 登录成功
            self.login_success.emit(username)
            self.clear_inputs()
        else:
            QMessageBox.warning(self, "登陆失败", '用户名或者密码不正确')
        # elif not os.path.exists("user.txt"):
        #     QMessageBox.warning(self,"登陆失败,还没有用户注册,需要先注册")
        #     return
        # elif not os.access("user.txt",os.R_OK):
        #     QMessageBox.information(self,"错误","文件不可读,重新输入")
        #     return
        # else:
        #     f = open("user.txt","r",encoding='utf-8')
        #     lines = f.readlines()
        #     f.close()
        #
        #     login_success = False
        #     for line in lines:
        #         line = line.strip()
        #         if not line:
        #             continue
        #
        #         parts = line.split(",")
        #         if len(parts)!=2:
        #             continue
        #
        #         stored_username,stored_password = parts#获取存储的账号密码
        #
        #         if stored_username == username and stored_password == password:
        #             login_success = True
        #             break
        #     if login_success:
        #         self.login_success.emit(username)
        #         # QMessageBox.information(self,"登录成功",f"欢迎回来,{username}!")
        #         self.clear_inputs()
        #     else:
        #         QMessageBox.warning(self,"登陆失败",'用户名或者密码不正确')
    def is_username_exists(self,username):
        #使用数据库
        query = QSqlQuery()
        query.prepare("SELECT username FROM users WHERE username = ?")
        query.addBindValue(username)

        if query.exec_() and query.next():
            return True
        return False





        # if not os.path.exists("user.txt"):
        #     return False
        # if not os.access("user.txt", os.R_OK):
        #     QMessageBox.critical(self, "错误", "文件不可读，无法检查用户名!")
        #     return False
        #     # 读取文件检查用户名
        # f = open(self.user_file, "r", encoding="utf-8")
        # lines = f.readlines()
        # f.close()

        # for line in lines:
        #     line = line.strip()
        #     if line and line.split(",")[0] == username:
        #         return True
        # return False


    def show_main_window(self,username):
        self.hide()#隐藏登录窗口
        self.main_window = MainWindow(username)#创建登陆后的主窗口
        self.main_window.show()
        #连接主窗口的退出信号
        self.main_window.logout_signal.connect(self.show_login_window)
        self.main_window.show()

    def show_login_window(self):
        """显示登录界面并隐藏主界面"""
        if self.main_window:
            self.main_window.hide()  # 隐藏主窗口
            self.main_window = None#初始化
        self.show()  # 显示登录窗口
        self.clear_inputs()  # 清空输入框


    def clear_inputs(self):
        self.username_edit.clear()
        self.password_edit.clear()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    sys.exit(app.exec_())



