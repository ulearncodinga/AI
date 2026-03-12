'''
1.界面设计：
  一个按钮：返回上一级目录
  一个单行文本框：显示当前目录的完整路径
 一个 QListWidget：显示当前目录下所有文件夹以及文件列表
  注意：在此采用了布局的方式设计界面，更符合实际
    按钮和文本框放在一个水平布局中 QHBoxLayout
    水平布局和 QListWidget 放在一个垂直布局中 QVBoxLayout
2.初始化当前目录到当行文本编辑框中
3.得到当前的目录路径
4.读取这个目录下所有文件夹以及文件信息列表
5.通过循环的方式给 QListWidget 中挨个添加文件夹以及文件信息（文件夹或者文件的名称）
6.选择列表中的选项双击：
  如果是文件夹，切换目录，重复3.4.5步骤
  如果是文件，就使用记事本的程序打开这个文件
   注意： python中，在一个程序中启动另外一个py程序，只能用命令的方式。
   格式： python test.py 文件路径

7.点击返回上一级目录按钮：
  如果能切换目录成功，重复3.4.5步骤
  否则：什么都不做
'''
import sys
from PyQt5.QtCore import QDir,QProcess
from PyQt5.QtGui import QIcon
from PyQt5.QtWidgets import QApplication,QWidget,QListWidget,QPushButton,QLabel,QMessageBox,QListWidgetItem
from file_browser import Ui_Form
class fileBrowser(QWidget,Ui_Form):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path_edit.setText("./")
        #QDir类是目录类的对象
        self.dir = QDir()
        #显示目录下的内容
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
            path = self.dir.absolutePath()#获取当前目录的绝对路径
            self.path_edit.setText(path)
            self.show_dir()
    def update_dir(self,item):
        '''
        打开选中的目录 或者 打开选中的文件
        :return:
        '''
        print(f"双击之后{item.text()}")
        result = self.dir.cd(item.text())
        if result:
            path = self.dir.absolutePath()#获取路径
            self.path_edit.setText(path)
            self.show_dir()
        else:
            filepath = self.dir.absolutePath() + "/" + item.text()
            self.process = QProcess(self)
            #命令
            argv=[]
            argv.append("notepad.py")
            argv.append(filepath)
            self.process.start("python",argv)#相当于 python notepade.py test.txt






    def show_dir(self):
        '''
        显示目录下的内容
        :return:
        '''
        #清空当前edit里的内容
        self.listWidget.clear()
        #获取当前目录(给dir设置目录)
        self.dir.setPath(self.path_edit.text())
        #过滤器:文件过滤器,排序过滤器
        Dir_filter = QDir.AllEntries | QDir.NoDotAndDotDot

        #排序过滤器 会先列出子目录后列出文件
        sortFiter = QDir.DirsFirst
        self.fileinfo_list = self.dir.entryInfoList(Dir_filter,sortFiter)
        # print(self.fileinfo_list)


        for i in self.fileinfo_list:
            #QListWidgetItem 作用:用来表示QListWidget中的项目,每一条QFileInfo都是一个列表项
            item = QListWidgetItem()
            if i.isDir():
                item.setIcon(QIcon("../images/icon/dir.png"))
            elif i.isFile():
                item.setIcon(QIcon("../images/icon/"))
            item.setText(i.fileName())
            self.listWidget.addItem(item)




        # print(f"当前路径{self.path_edit.text}")







if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = fileBrowser()
    w.show()
    sys.exit(app.exec_())






















