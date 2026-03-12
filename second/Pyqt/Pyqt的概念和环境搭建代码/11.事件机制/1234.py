from PyQt5.QtWidgets import QWidget, QApplication, QLineEdit, QVBoxLayout
from PyQt5.QtCore import QEvent
import sys


class FilterWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("事件过滤器：多个编辑框禁止输入")
        self.resize(300, 200)

        # 创建布局
        layout = QVBoxLayout()

        # 创建3个编辑框
        self.edit1 = QLineEdit(self)
        self.edit1.setPlaceholderText("编辑框1（禁止输入）")
        self.edit2 = QLineEdit(self)
        self.edit2.setPlaceholderText("编辑框2（禁止输入）")
        self.edit3 = QLineEdit(self)
        self.edit3.setPlaceholderText("编辑框3（禁止输入）")

        # 给3个编辑框安装事件过滤器（过滤器对象为当前窗口）
        self.edit1.installEventFilter(self)
        self.edit2.installEventFilter(self)
        self.edit3.installEventFilter(self)

        # 添加到布局
        layout.addWidget(self.edit1)
        layout.addWidget(self.edit2)
        layout.addWidget(self.edit3)
        self.setLayout(layout)

    # 重写事件过滤函数
    def eventFilter(self, obj, event):
        # 1. 判断目标控件是否为我们的编辑框
        if obj in [self.edit1, self.edit2, self.edit3]:
            # 2. 判断事件类型是否为键盘输入事件
            if event.type() == QEvent.KeyPress:
                # 禁止输入：返回True，事件不传递给编辑框
                return True
        # 其他事件/其他控件：返回False，事件正常传递
        return super().eventFilter(obj, event)


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = FilterWindow()
    window.show()
    sys.exit(app.exec_())