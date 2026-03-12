"""
图片的显示=>
    图标 动态图 静态图


"""
import sys
from PyQt5.QtGui import QIcon,QPixmap,QMovie
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QPushButton

from show_pic_ui import Ui_Form


class Pic_window(QWidget,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        #设置图标
        icon = QIcon()
        #QIcon.Normal或者QIcon.Disabled,表示图标在正常状态下和禁用状态下的显示方式
        #QIcon.Off,QIcon.on.图标在按钮下的状态和未按下状态的显示方式
        icon.addPixmap(QPixmap("../images/icon/logo.ico"),QIcon.Normal,QIcon.Off)
        self.setWindowIcon(icon)

        self.slot_init()

    def slot_init(self):
        self.btn_fan.clicked.connect(self.show_gif)
        self.btn_light.clicked.connect(self.show_pic)
    def show_pic(self):
        #scaled把图片缩放到标签的大小
        self.label_light.setPixmap(QPixmap("../images/light_on.jpg").scaled(self.label_light.size()))


    def show_gif(self):
        movie = QMovie()
        movie.setFileName("../images/fan.gif")
        #标签内容自适应
        self.label_fan.setScaledContents(True)
        #给标签设置动图
        self.label_fan.setMovie(movie)
        movie.start()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = Pic_window()
    window.show()
    sys.exit(app.exec())