'''
    实现音乐播放器
'''
import sys
from PyQt5.QtCore import Qt,QFileInfo,QUrl,QTimer,QTime
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtMultimedia import QMediaContent,QMediaPlayer
from PyQt5.QtWidgets import (QApplication,QWidget,QLabel,QHBoxLayout,QPushButton,
                             QVBoxLayout,QSlider,QComboBox,QListWidget,QListWidgetItem,QFileDialog)

class Video_Player(QWidget):
    def __init__(self):
        super().__init__()
        self.songname_index = 0
        self.ui_init()
        #设置音乐播放器

        self.slot_init()

    def ui_init(self):
        '''
        界面创建
        :return:
        '''

        self.MP3_Player = QMediaPlayer(self)
        #创建播放视频的窗口
        self.video_widget = QVideoWidget(self)
        #设置视频输出
        self.MP3_Player.setVideoOutput(self.video_widget)
        self.setWindowTitle("视频播放器")
        #创建歌曲列表
        self.song_list = QListWidget(self)
        self.song_list.setMaximumWidth(150)





        # self.MP3_Player.setVideoOutput(self.video_widget)

        #准备一个水平布局,放video
        self.video_hbox = QHBoxLayout()
        self.video_hbox.addWidget(self.song_list)
        self.video_hbox.addWidget(self.video_widget)




        #显示歌曲名称
        self.name_lab = QLabel("视频名称",self)
        #显示播放时间
        self.time_lab = QLabel("00:00/00:00",self)

        #创建水平布局
        self.label_layout = QHBoxLayout()
        self.label_layout.addWidget(self.name_lab)
        self.label_layout.addWidget(self.time_lab)

        #创建播放滑块
        self.player_slider = QSlider(self)
        self.player_slider.setValue(0)
        #设置水平方向
        self.player_slider.setOrientation(Qt.Horizontal)



        #设置按钮
        self.play_btn = QPushButton("播放",self)
        self.choose_btn = QPushButton('选择文件',self)
        self.pre_btn = QPushButton('上一首',self)
        self.next_btn = QPushButton('下一首',self)


        #设置下拉框
        self.play_combox = QComboBox(self)
        item_value = ["单曲循环","顺序播放","随机播放"]
        self.play_combox.addItems(item_value)
        #音量滑块
        self.volume_slider = QSlider(self)
        self.volume_slider.setRange(0,100)
        self.volume_slider.setValue(30)
        #滑块水平
        self.volume_slider.setOrientation(Qt.Horizontal)

        #设置按钮的水平
        self.btn_layout = QHBoxLayout()
        self.btn_layout.addWidget(self.play_btn)
        self.btn_layout.addWidget(self.choose_btn)
        self.btn_layout.addWidget(self.pre_btn)
        self.btn_layout.addWidget(self.next_btn)
        self.btn_layout.addWidget(self.play_combox)
        self.btn_layout.addWidget(self.volume_slider)

        #设置垂直布局(把四个全放在垂直布局)按顺序
        self.vbox = QVBoxLayout()

        self.vbox.addLayout(self.video_hbox)
        self.vbox.addLayout(self.label_layout)
        self.vbox.addWidget(self.player_slider)
        self.vbox.addLayout(self.btn_layout)

        #把垂直布局放在整个界面中
        self.setLayout(self.vbox)
        #创建媒体播放器的对象
        # self.MP3_Player = QMediaPlayer(self)

    def slot_init(self):
        '''
        槽函数
        :return:
        '''
        self.play_btn.clicked.connect(self.play_btn_slot)
        self.choose_btn.clicked.connect(self.choose_btn_slot)


        #滑块功能关联
        self.MP3_Player.positionChanged.connect(self.update_playslider_value)
        self.MP3_Player.durationChanged.connect(self.update_playslider_range)


        #依据滑块的位置更新歌曲播放的位置
        self.player_slider.sliderMoved.connect(self.update_player_position)

        #依据音量滑块的位置,更新音量大小
        self.volume_slider.sliderMoved.connect(self.update_player_volume )


        #更新播放时间显示的定时器
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_showtime_slot)


    def play_btn_slot(self):
        if self.play_btn.text() == '播放':
            self.timer.start(1000)
            self.MP3_Player.play()
            self.play_btn.setText('暂停')
        else:
            self.timer.stop()
            self.MP3_Player.pause()
            self.play_btn.setText("播放")


    def choose_btn_slot(self):
        #弹出文件对话框
        self.music_path = QFileDialog.getOpenFileName(self,'选择视频','./','视频(*.mp4 *.avi *.mkv)')[0]

        #将歌曲的名称放到歌曲列表中
        #获取歌曲名称
        self.music_name = QFileInfo(self.music_path).fileName()
        #吧歌曲名称添加到我们的列表中
        self.song_list.addItem(QListWidgetItem(self.music_name))
        self.name_lab.setText(self.music_name)
        #媒体播放器加载视频

        #视频画面的显示
        self.MP3_Player.setMedia(QMediaContent(QUrl.fromLocalFile(self.music_path)))



        #启动歌曲名称滚动显示得定时器功能
        self.startTimer(1000)




    def timerEvent(self,a0):
        '''
        歌曲名称滚动显示
        是一个虚函数,可以进行重写,以方便处理定时器事件
        :param a0:
        :return:
        '''
        if self.songname_index == len(self.music_name):
            self.songname_index =0
        else:
            self.songname_index += 1
            self.name_lab.setText(self.music_name[self.songname_index:])

    def update_showtime_slot(self):
        '''
        更新音乐播放时间
        :return:
        '''
        #首先来获取歌曲当前播放时长,返回一个int类型的时长,单位是ms
        cur_playtime = self.MP3_Player.position()

        #获取总时长
        music_time = self.MP3_Player.duration()


        #时间格式的转换,将int类型转换为(分钟:秒)的格式
        cur_playtime_str = QTime(0,0,0,0).addMSecs(cur_playtime).toString("mm:ss")
        music_time_str = QTime(0, 0, 0, 0).addMSecs(music_time).toString("mm:ss")

        self.time_lab.setText(cur_playtime_str+'/'+music_time_str)

    def update_playslider_value(self,position):
        '''
        更新滑块,播放滑块的当前位置
        :return:
        '''
        #获取到的当前时长设置给滑块
        self.player_slider.setValue(position)

    def update_playslider_range(self,duration):
        '''
        更新总时长
        :return:
        '''
        #总时长更新成我们歌曲的参数
        self.player_slider.setRange(0,duration)
    def update_player_position(self,position):
        '''
        滑块-->歌曲位置
        :return:
        '''
        self.MP3_Player.setPosition(position)

    def update_player_volume(self,position):
        self.MP3_Player.setVolume(position)



if __name__ == '__main__':
    app = QApplication(sys.argv)
    w = Video_Player()
    w.show()
    sys.exit(app.exec_())