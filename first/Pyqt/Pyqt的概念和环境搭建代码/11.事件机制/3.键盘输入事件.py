'''
keyPressEvent()
'''
import sys
from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import QApplication, QWidget
 

class KeyEvent(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("键盘输入事件")
    def keyPressEvent(self, a0):
        '''
        重写键盘输入事件处理函数
        :param a0: 是一个QKeyEvent对象
        :return:
        '''
        print(a0)
        key = a0.key()#返回的是按下键的键值
        print(f'key0 = {key}')

        text = a0.text()#返回的是对应的案件字符串
        print(f'text = {text}')

        modifiers = a0.modifiers()#返回键盘修饰键 比如CTRL,shift,alt
        print(f'modifiers = {modifiers}')


        if modifiers == Qt.ShiftModifier | Qt.ControlModifier:
            print('按下了shift + ctrl键')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = KeyEvent()
    w.show()
    sys.exit(app.exec_())