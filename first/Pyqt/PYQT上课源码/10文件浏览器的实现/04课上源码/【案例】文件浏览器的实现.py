"""
    实现文件浏览器
"""
import sys
from PyQt5.QtCore import QDir, QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication, QWidget, QListWidgetItem

from file_browser_ui import Ui_Form

class fileBrowser(QWidget, Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path_edit.setText("./")
        # QDir类是目录类的对象
        self.dir = QDir()
        # 显示目录下的内容
        self.show_dir()
        self.slot_init()

    def slot_init(self):
        """
        槽函数
        :return:
        """
        self.back_btn.clicked.connect(self.back_btn_slot)
        self.listWidget.itemDoubleClicked.connect(self.update_dir)

    def back_btn_slot(self):
        """
        返回上一级目录按钮  功能实现
        :return:
        """
        result = self.dir.cdUp()
        if result:
            path = self.dir.absolutePath()
            self.path_edit.setText(path)
            self.show_dir()

    def update_dir(self, item):
        """
        打开选中的目录或者打开选中的文件
        :return:
        """
        # print(f'双击之后{item.text()}')
        result = self.dir.cd(item.text())
        if result:
            path = self.dir.absolutePath()
            self.path_edit.setText(path)
            self.show_dir()
        else:
            filepath = self.dir.absolutePath() + "/" + item.text()
            self.process = QProcess(self)

            # 命令
            argv = []
            argv.append("notepad.py")
            argv.append(filepath)

            self.process.start("python", argv) # 相当于 python notepade.py test.txt

    def show_dir(self):
        """
        显示目录下的内容
        :return:
        """
        # 清空当前 edit 里面的内容
        self.listWidget.clear()
        # 给 dir 设置目录
        self.dir.setPath(self.path_edit.text())
        # 文件过滤器
        Dir_filter = QDir.AllEntries | QDir.NoDotAndDotDot

        # 排序过滤器  会先列出子目录后列出文件
        sortFilter = QDir.DirsFirst

        self.fileinfo_list = self.dir.entryInfoList(Dir_filter, sortFilter)

        for i in self.fileinfo_list:
            # QListWidgetItem 作用，用来表示QListWidget中的项目，每一条QFileInfo都是一个列表项
            item = QListWidgetItem()
            if i.isDir():
                item.setIcon(QIcon("../images/icon/dir.png"))
            elif i.isFile():
                item.setIcon(QIcon("../images/icon/file.png"))
            item.setText(i.fileName())
            self.listWidget.addItem(item)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = fileBrowser()
    w.show()
    sys.exit(app.exec_())
