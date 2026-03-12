'''
重写 : closeEvent()

再点击关闭窗口前,弹出是否确认关闭
'''
import sys
from PyQt5.QtWidgets import QWidget,QApplication,QMessageBox

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("关闭事件")

    def closeEvent(self, a0):
        result = QMessageBox.question(self,"提示","确定要关闭吗?",QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            #用来关闭窗口事件
            a0.accept()
        else:
            #忽略关闭事件
            a0.ignore()
if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
