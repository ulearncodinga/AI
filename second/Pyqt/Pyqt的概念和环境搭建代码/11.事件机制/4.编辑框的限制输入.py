import sys
from PyQt5.QtCore import Qt, QEvent
from PyQt5.QtWidgets import QApplication, QWidget, QLineEdit


class NumberOnlyLineEdit(QLineEdit):
    def __init__(self, parent=None):
        super().__init__(parent)
        # 设置输入掩码，只允许数字输入
        self.setInputMask("9" * 30)  # 允许最多30位数字
        self.setPlaceholderText("只能输入数字...")

    def keyPressEvent(self, event):
        # 允许功能键（退格、删除、方向键等）
        if event.key() in [
            Qt.Key_Backspace, Qt.Key_Delete,
            Qt.Key_Left, Qt.Key_Right,
            Qt.Key_Tab, Qt.Key_Return, Qt.Key_Enter
        ]:
            super().keyPressEvent(event)
            return

        # 检查输入是否为数字
        if event.text().isdigit():
            super().keyPressEvent(event)
        else:
            # 非数字输入则忽略
            event.ignore()

    # 防止通过粘贴输入非数字内容
    def insertFromMimeData(self, source):
        text = source.text()
        if text.isdigit():
            super().insertFromMimeData(source)


class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("严格限制只能输入数字")
        self.resize(300, 100)

        self.edit = NumberOnlyLineEdit(self)
        self.edit.setGeometry(50, 30, 200, 40)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())
