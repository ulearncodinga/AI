"""
    关闭事件
"""
import sys
from PyQt5.QtWidgets import QWidget, QApplication, QMessageBox


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("关闭事件")

    def closeEvent(self, a0):
        result = QMessageBox.question(self, "提示", "确定要关闭窗口吗？", QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            # 用来接收窗口关闭事件
            a0.accept()
        else:
            # 用来忽略关闭事件
            a0.ignore()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())