import sys
from PyQt5.QtWidgets import QApplication,QWidget,QLabel
from PyQt5.QtCore import QTime,QTimer,QDateTime

class Window(QWidget):
    def __init__(self):
        super().__init__()
        self.ui_init()

    def ui_init(self):
        self.setWindowTitle("PyQt5定时器")



        #使用label获取当前时间
        self.showtime_label = QLabel("PyQt定时器",self)
        self.showtime_label.setGeometry(100,50,130,30)


        #引入定时器
        self.timer = QTimer(self)
        #设置超时时间  1s 参数单位是毫秒
        self.timer.setInterval(1000)
        #timeout 定时器产生的信号
        self.timer.timeout.connect(self.update_time)
        #定时器需要启动,否则不工作
        self.timer.start()





    def update_time(self):
        '''
        获取当前时间
        :return:
        '''
        #时间格式:yyyy  四位年份   MM 两位月份   dd 2位天   hh  二十四小时制   mm 分钟  ss秒  zzz  毫秒
        self.current_time = QDateTime.currentDateTime().toString("hh:mm:ss")
        self.showtime_label.setText(self.current_time)












if __name__ == '__main__':

    app = QApplication(sys.argv)
    window = Window()
    window.show()
    sys.exit(app.exec())