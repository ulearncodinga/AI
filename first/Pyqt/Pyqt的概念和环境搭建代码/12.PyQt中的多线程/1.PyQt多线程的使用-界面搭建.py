'''
    PyQt多线程的使用
'''
import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QPushButton,QFrame,QMessageBox

class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt多线程")
        self.ui_init()
        self.slot_init()

    def ui_init(self):
        self.camera_lab = QLabel(self)
        self.camera_lab.setGeometry(100,50,300,300)
        self.camera_lab.setFrameShape(QFrame.Box)
        #setScaledContents方法时缩放内容使其填充到lab中可用的空间内
        self.camera_lab.setScaledContents(True)

        self.grayscale_lab = QLabel(self)
        self.grayscale_lab.setGeometry(500, 50, 300, 300)
        self.grayscale_lab.setFrameShape(QFrame.Box)
        # setScaledContents方法时缩放内容使其填充到lab中可用的空间内
        self.grayscale_lab.setScaledContents(True)
            #打开摄像头
        self.open_btn = QPushButton("打开摄像头",self)
        self.open_btn.setGeometry(200,400,100,30)
            #录制视频
        self.save_btn = QPushButton("录制视频", self)
        self.save_btn.setGeometry(600, 400, 100, 30)

    def slot_init(self):
        self.open_btn.clicked.connect(self.open_btn_slot)
        self.save_btn.clicked.connect(self.save_btn_slot)
    def open_btn_slot(self):
        '''
        打开摄像头,抓取画面数据
        :return:
        '''
        pass
    def save_btn_slot(self):
        '''
        开启子线程二:接收子线程一的图像数据,并完成视频的保存
        :return:
        '''
        pass


    def closeEvent(self, a0):
        result = QMessageBox.question(self,"提示","确定要关闭窗口吗?",QMessageBox.Yes | QMessageBox.No)
        if result == QMessageBox.Yes:
            #用来接收窗口关闭事件
            a0.accept()
        else:
            a0.ignore()

if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Widget()
    w.show()
    sys.exit(app.exec_())