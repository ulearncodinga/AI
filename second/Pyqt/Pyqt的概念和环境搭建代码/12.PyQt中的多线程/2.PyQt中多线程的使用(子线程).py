'''
    PyQt多线程的使用
'''
import sys
import cv2
import numpy as np
from PyQt5.QtCore import QThread,pyqtSignal,QDateTime
from PyQt5.QtGui import QImage,QPixmap
from PyQt5.QtWidgets import QApplication,QWidget,QLabel,QPushButton,QFrame,QMessageBox

class cameraThread(QThread):
    send_img = pyqtSignal(QImage,QImage)
    send_mat = pyqtSignal(np.ndarray)

    def __init__(self):
        super().__init__()


    def run(self):
        #打开摄像头
        cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
        if cap.isOpened():
            while True:
                #读取画面
                ret,frame = cap.read()
                if ret:
                    #对读取到的画面信息做数据处理,镜像处理
                    fst = cv2.flip(frame,1)
                    #BGR=>RGB
                    img_rgb = cv2.cvtColor(fst,cv2.COLOR_BGR2RGB)
                    #GR->GRAY
                    img_gray = cv2.cvtColor(fst,cv2.COLOR_BGR2GRAY)

                    #字节数组->QImage图像
                    rgb_img = QImage(img_rgb.data,img_rgb.shape[1],img_rgb.shape[0],QImage.Format_RGB888)
                    gray_img = QImage(img_gray.data, img_gray.shape[1], img_gray.shape[0], QImage.Format_Grayscale8)

                    #把RGB和灰度图发送给主线程
                    self.send_img.emit(rgb_img,gray_img)
                    #发送字节数组给子线程2
                    self.send_mat.emit(fst)


class saveThread(QThread):
    def __init__(self):
        super().__init__()
        self.count=0

    def run(self):
        #创建保存视频的对象
        writer = cv2.VideoWriter()
        filename = "./video/" + QDateTime.currentDateTime().toString("hh-mm-ss") + ".avi"
        #编解码器
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        result = writer.open(filename,fourcc,500,(640,480))
        if result:
            while True:
                if self.mat is not None:
                    #保存这一帧的图像
                    writer.write(self.mat)
                    self.count += 1
                    print(f"第{self.count}帧图像")
                    if self.count == 3000:
                        #关闭当前文件
                        writer.release()
                        filename = "./video/"+ QDateTime.currentDateTime().toString("hh-mm-ss")+".avi"
                        result = writer.open(filename,fourcc,500,(640,480))
                        self.count = 0
                else:
                    continue
        else:
            print("文件打开失败")


    def recv_mat(self,mat):
        self.mat = mat






class Widget(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("PyQt多线程")
        self.thread_1 = cameraThread()
        self.thread_2 = saveThread()
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

        self.thread_1.send_img.connect(self.show_img)
        self.thread_1.send_mat.connect(self.thread_2.recv_mat)
    def open_btn_slot(self):
        '''
        打开摄像头,抓取画面数据
        :return:
        '''
        self.thread_1.start()
    def save_btn_slot(self):
        '''
        开启子线程二:接收子线程一的图像数据,并完成视频的保存
        :return:
        '''
        self.thread_2.start()
    def show_img(self,rgb_img,gray_img):
        self.camera_lab.setPixmap(QPixmap.fromImage(rgb_img))
        self.grayscale_lab.setPixmap(QPixmap.fromImage(gray_img))
        self.update()

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