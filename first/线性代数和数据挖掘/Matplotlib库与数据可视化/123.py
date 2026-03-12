# 导入sys模块，用于处理Python解释器相关操作，如命令行参数、退出程序等
import sys
# 从PyQt5的QtWidgets模块导入所需的UI组件
from PyQt5.QtWidgets import (QApplication, QMainWindow, QStackedWidget,
                             QWidget, QPushButton, QVBoxLayout, QHBoxLayout,
                             QLabel, QLineEdit, QFormLayout, QMessageBox)
# 从PyQt5的QtCore模块导入核心功能组件，包括QObject基类、信号机制和Qt常量
from PyQt5.QtCore import QObject, pyqtSignal, Qt
# 从PyQt5的QtGui模块导入QFont类，用于处理字体相关设置
from PyQt5.QtGui import QFont


# 创建信号管理器类，继承自QObject以支持信号机制
class SignalManager(QObject):
    """信号管理器，集中管理应用中的所有自定义信号"""
    # 定义切换页面信号，参数为页面索引（整数类型）
    switch_page = pyqtSignal(int)
    # 定义用户登录信号，参数为用户名（字符串类型）
    user_login = pyqtSignal(str)
    # 定义更新设置信号，参数为设置字典（字典类型）
    update_settings = pyqtSignal(dict)


# 创建登录页面类，继承自QWidget作为基础窗口部件
class LoginPage(QWidget):
    """登录页面"""

    # 构造方法，接收信号管理器和父窗口作为参数
    def __init__(self, signal_manager, parent=None):
        # 调用父类QWidget的构造方法，确保正确初始化
        super().__init__(parent)
        # 保存信号管理器的引用，用于发射信号
        self.signal_manager = signal_manager
        # 初始化UI界面
        self.init_ui()

    # 初始化登录页面的UI元素
    def init_ui(self):
        # 创建垂直布局管理器，并应用于当前窗口
        layout = QVBoxLayout(self)
        # 设置布局中部件之间的间距为20像素
        layout.setSpacing(20)
        # 设置布局与窗口边缘的边距为50像素（上、右、下、左）
        layout.setContentsMargins(50, 50, 50, 50)

        # 创建标题标签，显示"用户登录"文本
        title_label = QLabel("用户登录")
        # 创建字体对象，用于设置标题字体
        title_font = QFont()
        # 设置字体大小为18点
        title_font.setPointSize(18)
        # 设置字体为粗体
        title_font.setBold(True)
        # 将设置好的字体应用到标题标签
        title_label.setFont(title_font)
        # 设置标题标签的文本对齐方式为居中对齐
        title_label.setAlignment(Qt.AlignCenter)
        # 将标题标签添加到布局中
        layout.addWidget(title_label)

        # 创建表单布局管理器，用于管理输入字段
        form_layout = QFormLayout()
        # 设置表单布局不自动换行
        form_layout.setRowWrapPolicy(QFormLayout.DontWrapRows)
        # 设置标签的对齐方式为右对齐且垂直居中
        form_layout.setLabelAlignment(Qt.AlignRight | Qt.AlignVCenter)
        # 设置输入字段的增长策略为自动扩展
        form_layout.setFieldGrowthPolicy(QFormLayout.ExpandingFieldsGrow)
        # 设置表单中部件之间的间距为15像素
        form_layout.setSpacing(15)

        # 创建用户名输入框
        self.username_edit = QLineEdit()
        # 创建密码输入框
        self.password_edit = QLineEdit()
        # 设置密码输入框的显示模式为密码模式（输入内容显示为圆点）
        self.password_edit.setEchoMode(QLineEdit.Password)

        # 向表单布局添加用户名行（标签+输入框）
        form_layout.addRow("用户名:", self.username_edit)
        # 向表单布局添加密码行（标签+输入框）
        form_layout.addRow("密码:", self.password_edit)

        # 将表单布局添加到主布局中
        layout.addLayout(form_layout)

        # 创建水平布局管理器，用于放置按钮
        btn_layout = QHBoxLayout()

        # 创建登录按钮，显示"登录"文本
        self.login_btn = QPushButton("登录")
        # 为登录按钮绑定点击事件处理函数on_login
        self.login_btn.clicked.connect(self.on_login)
        # 设置登录按钮的最小高度为30像素
        self.login_btn.setMinimumHeight(30)

        # 在按钮布局左侧添加伸缩项（用于右对齐按钮）
        btn_layout.addStretch()
        # 将登录按钮添加到按钮布局
        btn_layout.addWidget(self.login_btn)
        # 在按钮布局右侧添加伸缩项
        btn_layout.addStretch()

        # 将按钮布局添加到主布局
        layout.addLayout(btn_layout)
        # 在布局底部添加伸缩项（将所有内容向上推）
        layout.addStretch()

        # 设置当前窗口的样式表（CSS-like语法）
        self.setStyleSheet("""
            QPushButton {
                background-color: #4CAF50; /* 按钮背景色：绿色 */
                color: white; /* 按钮文本色：白色 */
                border-radius: 5px; /* 按钮圆角半径：5像素 */
                padding: 5px 15px; /* 按钮内边距：上下5px，左右15px */
                font-size: 14px; /* 按钮文本大小：14px */
            }
            QPushButton:hover {
                background-color: #45a049; /* 鼠标悬停时的按钮背景色：深绿色 */
            }
            QLineEdit {
                padding: 5px; /* 输入框内边距：5px */
                border: 1px solid #ccc; /* 输入框边框：1px灰色实线 */
                border-radius: 4px; /* 输入框圆角半径：4像素 */
                font-size: 14px; /* 输入框文本大小：14px */
            }
            QLabel {
                font-size: 14px; /* 标签文本大小：14px */
            }
        """)

    # 登录按钮点击事件的处理函数
    def on_login(self):
        """处理登录逻辑"""
        # 获取用户名输入框的文本并去除首尾空格
        username = self.username_edit.text().strip()
        # 获取密码输入框的文本并去除首尾空格
        password = self.password_edit.text().strip()

        # 简单验证（实际应用中应连接数据库验证）
        # 检查用户名是否为空
        if not username:
            # 显示警告消息框，提示用户输入用户名
            QMessageBox.warning(self, "警告", "请输入用户名")
            # 退出函数，不继续执行后续登录逻辑
            return

        # 检查密码是否为空
        if not password:
            # 显示警告消息框，提示用户输入密码
            QMessageBox.warning(self, "警告", "请输入密码")
            # 退出函数，不继续执行后续登录逻辑
            return

        # 这里简化处理，假设任何用户名密码组合都能登录
        # 发射用户登录信号，传递用户名为参数
        self.signal_manager.user_login.emit(username)
        # 发射切换页面信号，参数1表示切换到首页
        self.signal_manager.switch_page.emit(1)  # 切换到首页


# 创建首页类，继承自QWidget作为基础窗口部件
class HomePage(QWidget):
    """首页"""

    # 构造方法，接收信号管理器和父窗口作为参数
    def __init__(self, signal_manager, parent=None):
        # 调用父类QWidget的构造方法，确保正确初始化
        super().__init__(parent)
        # 保存信号管理器的引用，用于发射信号
        self.signal_manager = signal_manager
        # 初始化用户名，默认为"用户"
        self.username = "用户"
        # 初始化UI界面
        self.init_ui()
        # 连接信号与槽函数
        self.connect_signals()

    # 初始化首页的UI元素
    def init_ui(self):
        # 创建垂直布局管理器，并应用于当前窗口
        layout = QVBoxLayout(self)
        # 设置布局中部件之间的间距为20像素
        layout.setSpacing(20)
        # 设置布局与窗口边缘的边距为20像素（上、右、下、左）
        layout.setContentsMargins(20, 20, 20, 20)

        # 创建欢迎信息标签，显示欢迎文本
        self.welcome_label = QLabel(f"欢迎回来，{self.username}！")
        # 创建字体对象，用于设置欢迎信息字体
        welcome_font = QFont()
        # 设置字体大小为16点
        welcome_font.setPointSize(16)
        # 将设置好的字体应用到欢迎标签
        self.welcome_label.setFont(welcome_font)
        # 将欢迎标签添加到布局中
        layout.addWidget(self.welcome_label)

        # 创建功能说明标签，显示功能说明文本
        info_label = QLabel("这是应用的主页面，您可以：")
        # 设置标签字体为Arial，大小12px
        info_label.setFont(QFont("Arial", 12))
        # 将功能说明标签添加到布局中
        layout.addWidget(info_label)

        # 定义功能列表
        features = [
            "查看个人信息",
            "管理数据",
            "访问系统设置",
            "查看帮助文档"
        ]

        # 遍历功能列表，为每个功能创建标签
        for feature in features:
            # 创建功能标签，显示带项目符号的功能文本
            feature_label = QLabel(f"• {feature}")
            # 设置标签字体为Arial，大小11px
            feature_label.setFont(QFont("Arial", 11))
            # 将功能标签添加到布局中
            layout.addWidget(feature_label)

        # 在功能列表下方添加伸缩项（将内容向上推）
        layout.addStretch()

        # 创建水平布局管理器，用于放置按钮
        btn_layout = QHBoxLayout()

        # 创建系统设置按钮，显示"系统设置"文本
        self.setting_btn = QPushButton("系统设置")
        # 为系统设置按钮绑定点击事件处理函数go_to_settings
        self.setting_btn.clicked.connect(self.go_to_settings)
        # 设置按钮的最小高度为30像素
        self.setting_btn.setMinimumHeight(30)

        # 创建退出登录按钮，显示"退出登录"文本
        self.logout_btn = QPushButton("退出登录")
        # 为退出登录按钮绑定点击事件处理函数logout
        self.logout_btn.clicked.connect(self.logout)
        # 设置按钮的最小高度为30像素
        self.logout_btn.setMinimumHeight(30)

        # 将系统设置按钮添加到按钮布局
        btn_layout.addWidget(self.setting_btn)
        # 将退出登录按钮添加到按钮布局
        btn_layout.addWidget(self.logout_btn)

        # 将按钮布局添加到主布局
        layout.addLayout(btn_layout)

        # 设置当前窗口的样式表
        self.setStyleSheet("""
            QPushButton {
                background-color: #2196F3; /* 按钮背景色：蓝色 */
                color: white; /* 按钮文本色：白色 */
                border-radius: 5px; /* 按钮圆角半径：5像素 */
                padding: 5px 15px; /* 按钮内边距：上下5px，左右15px */
                font-size: 14px; /* 按钮文本大小：14px */
            }
            QPushButton:hover {
                background-color: #0b7dda; /* 鼠标悬停时的按钮背景色：深蓝色 */
            }
            QPushButton:nth-child(2) {
                background-color: #f44336; /* 第二个按钮背景色：红色 */
            }
            QPushButton:nth-child(2):hover {
                background-color: #d32f2f; /* 鼠标悬停时第二个按钮的背景色：深红色 */
            }
        """)

    # 连接信号与槽函数的方法
    def connect_signals(self):
        """连接信号"""
        # 将用户登录信号与更新用户名的槽函数关联
        self.signal_manager.user_login.connect(self.update_username)

    # 更新用户名显示的槽函数
    def update_username(self, username):
        """更新用户名显示"""
        # 更新用户名变量
        self.username = username
        # 更新欢迎标签的文本
        self.welcome_label.setText(f"欢迎回来，{self.username}！")

    # 前往设置页面的处理函数
    def go_to_settings(self):
        """前往设置页面"""
        # 发射切换页面信号，参数2表示切换到设置页面
        self.signal_manager.switch_page.emit(2)

    # 退出登录的处理函数
    def logout(self):
        """退出登录，返回登录页"""
        # 发射切换页面信号，参数0表示切换到登录页面
        self.signal_manager.switch_page.emit(0)
        # 清空输入（在登录页面中，实际应用中可能还需要清除用户状态等）


# 创建设置页面类，继承自QWidget作为基础窗口部件
class SettingsPage(QWidget):
    """设置页面"""

    # 构造方法，接收信号管理器和父窗口作为参数
    def __init__(self, signal_manager, parent=None):
        # 调用父类QWidget的构造方法，确保正确初始化
        super().__init__(parent)
        # 保存信号管理器的引用，用于发射信号
        self.signal_manager = signal_manager
        # 初始化设置字典，包含主题、通知和自动保存设置
        self.settings = {
            "theme": "light",  # 主题：浅色
            "notifications": True,  # 通知：开启
            "auto_save": False  # 自动保存：关闭
        }
        # 初始化UI界面
        self.init_ui()

    # 初始化设置页面的UI元素
    def init_ui(self):
        # 创建垂直布局管理器，并应用于当前窗口
        layout = QVBoxLayout(self)
        # 设置布局中部件之间的间距为20像素
        layout.setSpacing(20)
        # 设置布局与窗口边缘的边距为20像素（上、右、下、左）
        layout.setContentsMargins(20, 20, 20, 20)

        # 创建标题标签，显示"系统设置"文本
        title_label = QLabel("系统设置")
        # 创建字体对象，用于设置标题字体
        title_font = QFont()
        # 设置字体大小为16点
        title_font.setPointSize(16)
        # 设置字体为粗体
        title_font.setBold(True)
        # 将设置好的字体应用到标题标签
        title_label.setFont(title_font)
        # 将标题标签添加到布局中
        layout.addWidget(title_label)

        # 创建水平布局管理器，用于放置主题设置相关部件
        theme_layout = QHBoxLayout()
        # 创建主题标签，显示"主题:"文本
        theme_label = QLabel("主题:")
        # 创建浅色主题按钮，显示"浅色"文本
        self.light_theme_btn = QPushButton("浅色")
        # 创建深色主题按钮，显示"深色"文本
        self.dark_theme_btn = QPushButton("深色")

        # 根据当前主题设置按钮的选中状态
        self.light_theme_btn.setChecked(self.settings["theme"] == "light")
        self.dark_theme_btn.setChecked(self.settings["theme"] == "dark")

        # 设置按钮为可选中状态（切换按钮）
        self.light_theme_btn.setCheckable(True)
        self.dark_theme_btn.setCheckable(True)

        # 为浅色主题按钮绑定点击事件，调用set_theme方法并传入"light"参数
        self.light_theme_btn.clicked.connect(lambda: self.set_theme("light"))
        # 为深色主题按钮绑定点击事件，调用set_theme方法并传入"dark"参数
        self.dark_theme_btn.clicked.connect(lambda: self.set_theme("dark"))

        # 将主题标签添加到主题布局
        theme_layout.addWidget(theme_label)
        # 将浅色主题按钮添加到主题布局
        theme_layout.addWidget(self.light_theme_btn)
        # 将深色主题按钮添加到主题布局
        theme_layout.addWidget(self.dark_theme_btn)
        # 在主题布局右侧添加伸缩项（将内容向左推）
        theme_layout.addStretch()

        # 将主题布局添加到主布局
        layout.addLayout(theme_layout)

        # 创建通知设置按钮，根据当前通知状态显示不同文本
        self.notification_btn = QPushButton(
            "开启通知" if not self.settings["notifications"] else "关闭通知"
        )
        # 设置按钮为可选中状态
        self.notification_btn.setCheckable(True)
        # 根据当前通知设置设置按钮的选中状态
        self.notification_btn.setChecked(self.settings["notifications"])
        # 为通知按钮绑定点击事件处理函数toggle_notifications
        self.notification_btn.clicked.connect(self.toggle_notifications)
        # 将通知按钮添加到主布局
        layout.addWidget(self.notification_btn)

        # 创建自动保存设置按钮，根据当前自动保存状态显示不同文本
        self.auto_save_btn = QPushButton(
            "开启自动保存" if not self.settings["auto_save"] else "关闭自动保存"
        )
        # 设置按钮为可选中状态
        self.auto_save_btn.setCheckable(True)
        # 根据当前自动保存设置设置按钮的选中状态
        self.auto_save_btn.setChecked(self.settings["auto_save"])
        # 为自动保存按钮绑定点击事件处理函数toggle_auto_save
        self.auto_save_btn.clicked.connect(self.toggle_auto_save)
        # 将自动保存按钮添加到主布局
        layout.addWidget(self.auto_save_btn)

        # 在设置选项下方添加伸缩项（将内容向上推）
        layout.addStretch()

        # 创建水平布局管理器，用于放置底部按钮
        btn_layout = QHBoxLayout()

        # 创建保存设置按钮，显示"保存设置"文本
        self.save_btn = QPushButton("保存设置")
        # 为保存按钮绑定点击事件处理函数save_settings
        self.save_btn.clicked.connect(self.save_settings)
        # 设置按钮的最小高度为30像素
        self.save_btn.setMinimumHeight(30)

        # 创建返回首页按钮，显示"返回首页"文本
        self.back_btn = QPushButton("返回首页")
        # 为返回按钮绑定点击事件处理函数go_back
        self.back_btn.clicked.connect(self.go_back)
        # 设置按钮的最小高度为30像素
        self.back_btn.setMinimumHeight(30)

        # 将保存按钮添加到按钮布局
        btn_layout.addWidget(self.save_btn)
        # 将返回按钮添加到按钮布局
        btn_layout.addWidget(self.back_btn)

        # 将按钮布局添加到主布局
        layout.addLayout(btn_layout)

        # 根据当前主题更新界面样式
        self.update_theme()

    # 设置主题的方法
    def set_theme(self, theme):
        """设置主题"""
        # 更新设置字典中的主题值
        self.settings["theme"] = theme
        # 更新主题按钮的选中状态
        self.light_theme_btn.setChecked(theme == "light")
        self.dark_theme_btn.setChecked(theme == "dark")
        # 应用主题样式
        self.update_theme()

    # 切换通知状态的方法
    def toggle_notifications(self):
        """切换通知状态"""
        # 取反当前通知状态
        self.settings["notifications"] = not self.settings["notifications"]
        # 更新通知按钮的显示文本
        self.notification_btn.setText(
            "开启通知" if not self.settings["notifications"] else "关闭通知"
        )

    # 切换自动保存状态的方法
    def toggle_auto_save(self):
        """切换自动保存状态"""
        # 取反当前自动保存状态
        self.settings["auto_save"] = not self.settings["auto_save"]
        # 更新自动保存按钮的显示文本
        self.auto_save_btn.setText(
            "开启自动保存" if not self.settings["auto_save"] else "关闭自动保存"
        )

    # 保存设置的方法
    def save_settings(self):
        """保存设置并发射信号"""
        # 发射更新设置信号，传递当前设置字典
        self.signal_manager.update_settings.emit(self.settings)
        # 显示信息消息框，提示设置已保存
        QMessageBox.information(self, "提示", "设置已保存")

    # 返回首页的方法
    def go_back(self):
        """返回首页"""
        # 发射切换页面信号，参数1表示切换到首页
        self.signal_manager.switch_page.emit(1)

    # 更新主题样式的方法
    def update_theme(self):
        """更新主题样式"""
        # 如果是深色主题
        if self.settings["theme"] == "dark":
            # 设置深色主题的样式表
            self.setStyleSheet("""
                QWidget {
                    background-color: #333; /* 窗口背景色：深灰色 */
                    color: white; /* 文本色：白色 */
                }
                QPushButton {
                    background-color: #555; /* 按钮背景色：中灰色 */
                    color: white; /* 按钮文本色：白色 */
                    border-radius: 5px; /* 按钮圆角半径：5像素 */
                    padding: 5px 15px; /* 按钮内边距：上下5px，左右15px */
                    font-size: 14px; /* 按钮文本大小：14px */
                }
                QPushButton:checked {
                    background-color: #2196F3; /* 选中状态的按钮背景色：蓝色 */
                }
            """)
        # 如果是浅色主题
        else:
            # 设置浅色主题的样式表
            self.setStyleSheet("""
                QWidget {
                    background-color: #fff; /* 窗口背景色：白色 */
                    color: #333; /* 文本色：深灰色 */
                }
                QPushButton {
                    background-color: #eee; /* 按钮背景色：浅灰色 */
                    color: #333; /* 按钮文本色：深灰色 */
                    border-radius: 5px; /* 按钮圆角半径：5像素 */
                    padding: 5px 15px; /* 按钮内边距：上下5px，左右15px */
                    font-size: 14px; /* 按钮文本大小：14px */
                }
                QPushButton:checked {
                    background-color: #2196F3; /* 选中状态的按钮背景色：蓝色 */
                    color: white; /* 选中状态的按钮文本色：白色 */
                }
            """)


# 创建主窗口类，继承自QMainWindow作为应用的主窗口
class MainWindow(QMainWindow):
    """主窗口，管理所有页面"""

    # 构造方法
    def __init__(self):
        # 调用父类QMainWindow的构造方法，确保正确初始化
        super().__init__()
        # 初始化UI界面
        self.init_ui()

    # 初始化主窗口的UI元素
    def init_ui(self):
        # 设置窗口标题
        self.setWindowTitle("PyQt多页面应用示例")
        # 设置窗口的位置和大小：x=100, y=100, 宽=800, 高=600
        self.setGeometry(100, 100, 800, 600)

        # 创建信号管理器实例，用于管理应用中的所有信号
        self.signal_manager = SignalManager()

        # 创建QStackedWidget作为页面容器，用于管理多个页面的切换
        self.stack = QStackedWidget()

        # 创建各个页面的实例，并传入信号管理器
        self.login_page = LoginPage(self.signal_manager)
        self.home_page = HomePage(self.signal_manager)
        self.settings_page = SettingsPage(self.signal_manager)

        # 将各个页面添加到页面容器中
        self.stack.addWidget(self.login_page)  # 索引0：登录页面
        self.stack.addWidget(self.home_page)   # 索引1：首页
        self.stack.addWidget(self.settings_page)  # 索引2：设置页面

        # 将页面容器设置为主窗口的中央部件
        self.setCentralWidget(self.stack)

        # 连接所有信号与槽函数
        self.connect_signals()

    # 连接信号与槽函数的方法
    def connect_signals(self):
        """连接所有信号与槽"""
        # 将切换页面信号与页面容器的setCurrentIndex方法关联，用于切换显示的页面
        self.signal_manager.switch_page.connect(self.stack.setCurrentIndex)

        # 将更新设置信号与处理设置更新的槽函数关联
        self.signal_manager.update_settings.connect(self.handle_settings_update)

    # 处理设置更新的槽函数
    def handle_settings_update(self, settings):
        """处理设置更新"""
        # 在控制台打印更新后的设置（实际应用中可以在这里处理设置变更）
        print(f"设置已更新: {settings}")
        # 实际应用中可以在这里应用设置变更


# 程序入口点
if __name__ == "__main__":
    # 创建QApplication实例，管理应用程序的生命周期和资源
    app = QApplication(sys.argv)
    # 创建主窗口实例
    window = MainWindow()
    # 显示主窗口
    window.show()
    # 进入应用程序的主事件循环，等待用户交互并处理事件
    sys.exit(app.exec_())