#新建、打开、保存、另存为、字体设置、颜色设置、复制、粘贴、剪切、恢复、撤销等
'''
《1》给记事本窗口设置图标
《2》给记事本窗口设置标题
《3》给记事本添加菜单栏、工具栏、状态栏
《4》给记事本菜单栏中添加菜单
《5》给记事本菜单栏中的菜单添加动作(选项)
《6》给记事本添加文本编辑区域
'''
import sys
from PyQt5.QtCore import QFile,QFileInfo,QIODevice

from PyQt5.QtWidgets import (QMainWindow,QWidget,QApplication,QMenu,QMenuBar,QAction,
                                QColorDialog,QToolBar,QFileDialog,QToolButton,QFontDialog,QTextEdit,QMessageBox)
from PyQt5.QtGui import QIcon,QKeySequence,QColor

class My_MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()

        self.file_name = "无标题-记事本"

        self.file_path = ""

        self.ui_init()
        self.slot_init()
        self.resize(400,500)


    def ui_init(self):
        self.setWindowTitle(self.file_name)
        self.setWindowIcon(QIcon("../images/icon/1.png"))


        #添加菜单栏  QMenubar
        self.menu_bar = QMenuBar(self)
        self.setMenuBar(self.menu_bar)


        #菜单栏中添加菜单 QMenu
        self.file_menu = QMenu("文件(F)",self)
        #将文件菜单添加到菜单栏中
        self.menu_bar.addMenu(self.file_menu)

        self.edit_menu = QMenu("编辑(E)",self)
        self.menu_bar.addMenu(self.edit_menu)

        #在菜单里去添加动作,同时给动作设计图标
        self.new_action = QAction("新建(N)",self)
        self.new_action.setIcon(QIcon("../images/icon/new.png"))
        #设计快捷方式
        self.new_action.setShortcut(QKeySequence.New)
        # self.new_action.setShortcut(QKeySequence("Ctrl+N"))#自定义快捷键
        #将动作添加到我们的菜单中
        self.file_menu.addAction(self.new_action)

        #打开动作
        self.open_action = QAction(QIcon("../images/icon/open.png"),"打开(O)",self)
        self.open_action.setShortcut(QKeySequence.Open)
        self.file_menu.addAction(self.open_action)

        #保存动作
        self.save_action = QAction(QIcon("../images/icon/save.png"),"保存(S)",self)
        self.save_action.setShortcut(QKeySequence.Save)
        self.file_menu.addAction(self.save_action)

        #另存为
        self.saveAs_action = QAction(QIcon("../images/icon/save.png"),"另存为(A)",self)
        self.saveAs_action.setShortcut(QKeySequence.SaveAs)
        self.file_menu.addAction(self.saveAs_action)

        # 在编辑菜单中添加动作
        # 复制
        self.copy_action = QAction(QIcon("../images/icon/copy.png"), "复制(C)", self)
        self.copy_action.setShortcut(QKeySequence.Copy)
        self.edit_menu.addAction(self.copy_action)

        # 粘贴
        self.paste_action = QAction(QIcon("../images/icon/paste.png"), "粘贴(P)", self)
        self.paste_action.setShortcut(QKeySequence.Paste)
        self.edit_menu.addAction(self.paste_action)

        # 剪切
        self.cut_action = QAction(QIcon("../images/icon/cut.png"), "剪切(C)", self)
        self.cut_action.setShortcut(QKeySequence.Cut)
        self.edit_menu.addAction(self.cut_action)

        # 撤销
        self.undo_action = QAction(QIcon("../images/icon/undo.png"), "撤销(U)", self)
        self.undo_action.setShortcut(QKeySequence.Undo)
        self.edit_menu.addAction(self.undo_action)

        # 恢复
        self.redo_action = QAction(QIcon("../images/icon/redo.png"), "恢复(R)", self)
        self.redo_action.setShortcut(QKeySequence.Redo)
        self.edit_menu.addAction(self.redo_action)





        #添加工具栏
        self.tool_bar = QToolBar(self)
        self.addToolBar(self.tool_bar)


        #将新建和保存两个动作添加到工具栏
        self.tool_bar.addAction(self.new_action)
        self.tool_bar.addAction(self.save_action)

        #字体
        self.font_btn = QToolButton(self)
        self.font_btn.setIcon(QIcon("../images/icon/font.png"))
        self.tool_bar.addWidget(self.font_btn)

        #颜色
        self.color_btn = QToolButton(self)
        self.color_btn.setIcon(QIcon("../images/icon/color.png"))
        self.tool_bar.addWidget(self.color_btn)



        #记事本写内容的区域 QTextEdit
        self.text_edit = QTextEdit(self)
        self.setCentralWidget(self.text_edit)





    def slot_init(self):
        self.new_action.triggered.connect(self.new_action_slot)#新建

        self.open_action.triggered.connect(self.open_action_slot)#打开

        self.save_action.triggered.connect(self.save_action_slot)#保存

        self.saveAs_action.triggered.connect(self.saveAs_action_slot)

        self.text_edit.textChanged.connect(self.update_title)

        #编辑菜单下的信号槽与函数的连接
        self.copy_action.triggered.connect(self.text_edit.copy)
        self.paste_action.triggered.connect(self.text_edit.paste)
        self.cut_action.triggered.connect(self.text_edit.cut)
        self.undo_action.triggered.connect(self.text_edit.undo)
        self.redo_action.triggered.connect(self.text_edit.redo)


        self.font_btn.clicked.connect(self.font_slot)
        self.color_btn.clicked.connect(self.color_slot)

    def new_action_slot(self):
        #QTextEdit内部有一个QTextDocument对象,用于存储和处理文本内容
        #document的作用可以获取到文本框中的文本文档,   isModified可以得到QTextDocument对象是否被修改
        result = self.text_edit.document().isModified()
        if result:
            btn_res = QMessageBox.question(self,"记事本","是否保存当前修改内容",QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)

            if btn_res == QMessageBox.Yes:

                #执行保存函数
                self.save_action_slot()
                #保存完成之后,把当前的文本框的内容清空
                self.text_edit.clear()
                self.file_name="无标题-记事本"
                self.setWindowTitle(self.file_name)

            elif btn_res == QMessageBox.No:
                self.text_edit.clear()
                self.file_name = "无标题-记事本"
                self.setWindowTitle(self.file_name)
            else:
                pass

    def open_action_slot(self):
        result =self.text_edit.document().isModified()
        if result:
            btn_res = QMessageBox.question(self,"记事本","是否保存当前内容",QMessageBox.Yes|QMessageBox.No|QMessageBox.Cancel)
            if btn_res == QMessageBox.Yes:
                self.save_action_slot()
                self.open_file()
            elif btn_res == QMessageBox.No:
                self.open_file()
        else:
            self.open_file()

    def open_file(self):
        # QFileDialog.getOpenFileName用于打开文件对话框,让用户选择文件的方法

        # "打开":对话框的标题  "./":代表当前目录  "记事本TXT(*.txt)文件过滤器 在对话框中只显示 *.txt文件类型"
        self.file_path = QFileDialog.getOpenFileName(self,"打开","./","记事本TXT(*.txt)")[0]
        print(self.file_path)


        fs = QFile(self.file_path)
        fs.open(QFile.ReadOnly)
        if fs.isOpen():
            # 读取文件内容,读取得到的是bytearray 则会使Python的字节数组
            data = fs.readAll()
            #将bytearray解码,解码之后放到edit
            self.text_edit.setText(bytearray(data).decode('utf-8'))#注意,必须保证是utf-8
            self.file_name = QFileInfo(self.file_path).fileName()
            self.setWindowTitle(self.file_name)
            fs.close()



        else:
            print("文件未打开")


    def save_action_slot(self):
        if self.file_name == "无标题 - 记事本":
            #另存为
            self.saveAs_action_slot()
        else:
            fs = QFile(self.file_path)
            fs.open(QFile.WriteOnly)
            if fs.isOpen():
                #获取到当前文本框的内容
                text = self.text_edit.toPlainText()
                fs.write(text.encode('utf-8'))
                self.file_name = QFileInfo(self.file_path).fileName()
                self.setWindowTitle(self.file_name)
                fs.close()
                self.text_edit.document().setModified(False)
            else:
                print('文件保存失败')



    def saveAs_action_slot(self):
        self.file_path = QFileDialog.getSaveFileName(self,"另存为",'./',"记事本TXT(*.txt)")[0]
        fs = QFile(self.file_path)
        fs.open(QFile.WriteOnly)
        if fs.isOpen():
            text = self.text_edit.toPlainText()
            fs.write(text.encode('utf-8'))
            self.file_name = QFileInfo(self.file_path).fileName()
            self.setWindowTitle(self.file_name)
            fs.close()
            self.text_edit.document().setModified(False)
        else:
            print('文件另存为失败')







    def font_slot(self):
        font, ok = QFontDialog.getFont(self.text_edit.font(),self)
        if ok:
            self.text_edit.setCurrentFont(font)



    def color_slot(self):
        #会返回一个QColor对象,表示用户选择的颜色
        color = QColorDialog.getColor(QColor(),self)
        if color.isValid:
            self.text_edit.setTextColor(color)




    def update_title(self):
        if self.text_edit.document().isModified():
            self.setWindowTitle("*" + self.file_name)
        else:
            self.setWindowTitle(self.file_name)



































if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = My_MainWindow()
    window.show()
    sys.exit(app.exec_())