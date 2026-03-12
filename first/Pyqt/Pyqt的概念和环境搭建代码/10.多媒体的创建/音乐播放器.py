'''
    实现一首音乐的播放
'''
import sys
from PyQt5.QtMultimedia import QMediaPlayer,QMediaContent
from PyQt5.QtCore import QUrl

from PyQt5.QtWidgets import QApplication,QWidget,QPushButton
class AudioPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("音乐播放")

        self.btn = QPushButton("播放/关闭",self)
        self.btn.setGeometry(100,50,100,40)

        #创建音频播放器
        self.player = QMediaPlayer()
        #QUrl.fromLocalFile将本地路径转化为 QMediaContent 可以使用的路径
        self.player.setMedia(QMediaContent(QUrl.fromLocalFile("./music/凤凰传奇 - 最炫民族风.mp3")))


        self.btn.clicked.connect(self.btn_slot)


    def btn_slot(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()#暂停
        else:
            self.player.play()


if __name__ == '__main__':

    app = QApplication(sys.argv)
    w = AudioPlayer()
    w.show()
    sys.exit(app.exec_())