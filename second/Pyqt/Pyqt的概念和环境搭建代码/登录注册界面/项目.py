import os.path
import sys
from PyQt5 import QtGui, QtCore
from PyQt5.QtGui import QIcon, QPixmap, QMovie
from PyQt5.QtCore import pyqtSignal
from PyQt5.QtSql import QSqlQuery, QSqlDatabase
from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QLineEdit, QPushButton, QMessageBox


class MySql:
    def __init__(self):
        '''
        连接数据库
        '''
        try:
            self.database = QSqlDatabase.addDatabase("QSQLITE")
            self.database.setDatabaseName("users2.db")
            self.database.open()
        except Exception as e:
            print(f"无法连接,error:{e}")

    def operation_sql(self, sql):
        self.query = QSqlQuery()#创建对象
        self.query.exec_(sql)#发送SQl语句

    def selectData(self, sql):
        '''
        查询并返回结果列表
        :return: 结果列表
        '''
        self.query = QSqlQuery()
        self.query.exec_(sql)
        self.result = self.query.record()
        result_list = []
        while self.query.next():
            row = []
            for i in range(self.result.count()):
                row.append(self.query.value(i))
            result_list.append(row)
        return result_list

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
        self.resize(800, 600)

        self.product_btn = QPushButton('商品管理', self)
        self.product_btn.setGeometry(50, 100, 150, 50)

        self.order_btn = QPushButton("订单管理", self)
        self.order_btn.setGeometry(50, 170, 150, 50)

        self.user_btn = QPushButton("用户管理", self)
        self.user_btn.setGeometry(50, 240, 150, 50)

        self.logout_btn = QPushButton("退出登录", self)
        self.logout_btn.setGeometry(50, 310, 150, 50)

        # 连接信号和槽
        self.logout_btn.clicked.connect(self.logout)

    def logout(self):
        self.logout_signal.emit()
        self.close()


class MyWindow(QWidget):
    login_success = pyqtSignal(str)

    def __init__(self,db):
        super().__init__()

        # 初始化数据库连接
        self.db = db

        # 使用MySql类直接执行SQL创建表
        # self.db.operation_sql("""
        #     CREATE TABLE IF NOT EXISTS reg_info (
        #         username VARCHAR(20) UNIQUE NOT NULL,
        #         password VARCHAR(20) NOT NULL
        #     )
        # """)

        # 可选：初始化测试数据
        # self.db.operation_sql("delete from users where username = 'test1' or username = 'test2'")
        # self.db.operation_sql("insert into users values('test1', '123456')")
        # self.db.operation_sql("insert into users values('test2', '654321')")

        # 窗口设置
        self.resize(600, 400)
        self.main_window = None
        self.ui_init()
        self.slot_init()
        self.setWindowTitle("购物管理系统")
        self.login_success.connect(self.show_main_window)
        self.set_background_image()

        # 界面图标按钮
        self.btn = QPushButton(self)
        self.btn.setGeometry(250, 20, 100, 50)
        self.btn.setStyleSheet("border-image:url(../images/HQYJ.png)")

        self.btnusername = QPushButton(self)
        self.btnusername.setGeometry(150, 80, 40, 40)
        self.btnusername.setStyleSheet("border-image:url(../images/OIP-C.webp)")

        self.btnpassword = QPushButton(self)
        self.btnpassword.setGeometry(150, 140, 40, 40)
        self.btnpassword.setStyleSheet("border-image:url(../images/password.webp)")

    def set_background_image(self):
        """设置背景图片并使其居中显示"""
        self.background_label = QLabel(self)
        self.background_label.setGeometry(0, 0, self.width(), self.height())

        pixmap = QPixmap("../images/background.png")
        if not pixmap.isNull():
            scaled_pixmap = pixmap.scaled(self.background_label.size())
            self.background_label.setPixmap(scaled_pixmap)
            self.background_label.setAlignment(QtCore.Qt.AlignCenter)

        self.background_label.lower()

    def resizeEvent(self, event):
        if hasattr(self, 'background_label'):
            self.background_label.setGeometry(0, 0, self.width(), self.height())
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
        # 用户名
        self.username = QLabel('用户名', self)
        self.username.setGeometry(100, 80, 100, 40)
        self.username_edit = QLineEdit(self)
        self.username_edit.setGeometry(200, 80, 250, 40)

        self.username_edit.setMaxLength(8)
        reg_exp = QtCore.QRegExp("^[a-zA-Z0-9]{1,8}$")
        reg_exp_validator = QtGui.QRegExpValidator(reg_exp, self.username_edit)
        self.username_edit.setValidator(reg_exp_validator)

        # 密码设置
        self.password = QLabel('密码', self)
        self.password.setGeometry(100, 140, 100, 40)
        self.password_edit = QLineEdit(self)
        self.password_edit.setGeometry(200, 140, 250, 40)

        self.password_edit.setMaxLength(8)
        self.password_edit.setEchoMode(QLineEdit.Password)

        # 登录按钮
        self.center_btn = QPushButton("登录", self)
        self.center_btn.setGeometry(150, 220, 120, 50)
        # 注册按钮
        self.reg_btn = QPushButton("注册", self)
        self.reg_btn.setGeometry(330, 220, 120, 50)

    def slot_init(self):
        self.reg_btn.clicked.connect(self.reg_slot)
        self.center_btn.clicked.connect(self.center_slot)

    def reg_slot(self):
        """注册按钮关联的槽函数"""
        username = self.username_edit.text()
        password = self.password_edit.text()

        if username == "" or password == "":
            QMessageBox.warning(self, "注册", "用户名或者密码不能为空")
        elif 0 < len(username) < 8:
            QMessageBox.warning(self, "注册", "用户名输入长度不符合要求")
        elif 0 < len(password) < 8:
            QMessageBox.warning(self, "注册", "密码输入长度不符合要求")
        else:
            if self.is_username_exists(username):
                QMessageBox.warning(self, "注册", "用户名已存在，请选择其他用户名")
                return

            # 使用MySql类执行插入操作
            self.db.operation_sql(f"INSERT INTO users (username, password) VALUES ('{username}', '{password}')")
            QMessageBox.information(self, "注册", "恭喜注册完成!")

    def center_slot(self):
        # 获取输入的用户名和密码
        username = self.username_edit.text().strip()
        password = self.password_edit.text().strip()

        if username == "" or password == "":
            QMessageBox.warning(self, "登录失败", "用户名或密码不能为空！")
            return

        # 使用MySql类查询数据
        result = self.db.selectData(f"SELECT * FROM users WHERE username = '{username}' AND password = '{password}'")

        if result:
            self.login_success.emit(username)
            self.clear_inputs()
        else:
            QMessageBox.warning(self, "登陆失败", '用户名或者密码不正确')

    def is_username_exists(self, username):
        # 使用MySql类查询用户名是否存在
        result = self.db.selectData(f"SELECT username FROM users WHERE username = '{username}'")
        return len(result) > 0

    def show_main_window(self, username):
        self.hide()
        self.main_window = MainWindow(username)
        self.main_window.show()
        self.main_window.logout_signal.connect(self.show_login_window)

    def show_login_window(self):
        if self.main_window:
            self.main_window.hide()
            self.main_window = None
        self.show()
        self.clear_inputs()

    def clear_inputs(self):
        self.username_edit.clear()
        self.password_edit.clear()

    def closeEvent(self, event):
        """关闭窗口时关闭数据库连接"""
        self.db.close_db()
        event.accept()


if __name__ == '__main__':
    my_sql = MySql()
    my_sql.operation_sql("create table if not exists users(ID integer primary key AUTOINCREMENT, username text, password text) ")
    # my_sql.operation_sql("delete from users where username in ('test1', 'test2')")  # 清理旧测试数据
    # my_sql.operation_sql("insert into users (username, password) values('test1', '123456')")  # 插入测试用户

    app = QApplication(sys.argv)
    window = MyWindow(my_sql)
    window.show()
    sys.exit(app.exec_())
