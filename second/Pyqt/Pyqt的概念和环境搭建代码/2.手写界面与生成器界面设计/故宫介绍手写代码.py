# # #发送者.信号.connect(槽函数)
# # 需求： 点击一个故宫按钮，在文本框中显示故宫的介绍！
# # 介绍内容：故宫建筑群以中轴线为中心，布局严谨有序，体现了中国古代建筑的艺术风格和哲学思想。屋顶上
# # 铺着黄色的琉璃瓦，在阳光下闪闪发光，显得气势磅礴；红色的墙面上，雕刻着各种图案，细节精致。故宫的
# # 建筑风格典雅瑰丽，展现了中国古代建筑的高超技艺和独特魅力。
# # 需要的组件： 一个按钮 QPushButton，一个文本浏览器 QTextBrowser
# # 实现方式：手写代码和使用Designer工具完成开发
# # '''
# import sys
# from PyQt5.QtWidgets import QApplication,QWidget,QPushButton,QTextBrowser
#
# class Widget(QWidget):
#     '''
#     继承于QWidget,提供了基础的窗口和组件
#     '''
#     def __init__(self):
#         """
#         构造函数
#         """
#         #super由于继承父类,不用参数就能找到临时的父类
#         super().__init__()
#         #UI界面的初始化
#         self.ui_init()
#         #窗口的标题
#         self.setWindowTitle("故宫的介绍")
#         #设置窗口的大小
#         self.resize(450,300)
#         #设置窗口的位置
#         self.move(400,400)
#
#         self.slot_init()
#
#
#     def ui_init(self):
#         #创建需要的组件
#         #创建一个按钮,self的作用:指定其父组件为当前的窗口
#         self.show_btn = QPushButton(self)
#         #给按钮起一个名字,设置按钮上的文本
#         self.show_btn.setText('故宫')
#
#         # 设置按钮的位置,setGEometry()的单个的调用中,可以单独的设置组件的位置以及大小
#         self.show_btn.setGeometry(10, 10, 100, 30)
#         # 创建文本框组件,并指定其父组件为当前窗口
#         self.show_txt = QTextBrowser(self)
#         # 移动文本框的位置
#         self.show_txt.setGeometry(200, 40, 200, 200)
#
#     def slot_init(self):
#         self.show_btn.clicked.connect(self.show_info)
#     def show_info(self):
#         #槽函数的作用:显示故宫的介绍
#         self.show_txt.setText(
#                             "故宫建筑群以中轴线为中心，布局严谨有序，体现了中国古代建筑的艺术风格和哲学思想。屋顶上"
#                             "铺着黄色的琉璃瓦，在阳光下闪闪发光，显得气势磅礴；红色的墙面上，雕刻着各种图案，细节精致。故宫的"
#                             "建筑风格典雅瑰丽，展现了中国古代建筑的高超技艺和独特魅力")
# if __name__ == '__main__':
#     app = QApplication(sys.argv)
#     window = Widget()
#     window.show()
#     app.exec()
# '''
#
# #
# #
# #
# #
# #
# #
# #
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
#
# import sys
# from PyQt5.QtWidgets import QWidget,QApplication,QTextBrowser,QPushButton
#
# class Widget(QWidget):
#     def __init__(self):
#         super().__init__()
#         self.ui_init()
#         self.setWindowTitle("故宫")
#         self.resize(450,300)
#         self.move(400,400)
#         self.slot_init()
#     def ui_init(self):
#         self.show_btn = QPushButton(self)
#         self.show_btn.setText("点击")
#         self.show_btn.setGeometry(10,10,100,30)
#         self.show_txt = QTextBrowser(self)
#         self.show_txt.setGeometry(200,40,200,200)
#     def slot_init(self):
#         self.show_btn.clicked.connect(self.show_info)
#     def show_info(self):
#         self.show_txt.setText("故宫很美,以下省略")
#
# if __name__ == '__main__':
#
#     app=QApplication(sys.argv)
#     window = Widget()
#     window.show()
#     app.exec()


import sys
from PyQt5.QtWidgets import QWidget,QApplication,QTextBrowser,QPushButton

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()
        self.setWindowTitle("故宫")
        self.resize(450,300)
        self.move(400,400)
        self.slot_init()
    def ui_init(self):
        self.show_btn = QPushButton(self)
        self.show_btn.setText("点击")
        self.show_btn.setGeometry(10,10,100,30)
        self.show_txt = QTextBrowser(self)
        self.show_txt.setGeometry(200,40,200,200)
    def slot_init(self):
        self.show_btn.clicked.connect(self.show_info)
    def show_info(self):
        self.show_txt.setText("故宫很美丽")
if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Widget()
    window.show()
    app.exec()







