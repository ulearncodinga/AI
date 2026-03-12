import sys
import os
from PyQt5.QtGui import QIcon
from PyQt5.QtCore import QUrl, Qt, QFileInfo
from PyQt5.QtMultimedia import QMediaPlayer, QMediaContent
from PyQt5.QtMultimediaWidgets import QVideoWidget
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QFrame,
                             QPushButton, QHBoxLayout, QLabel, QSlider,
                             QFileDialog, QMessageBox)

"""错误日志重定向"""
import traceback
import warnings

# 屏蔽 deprecated 警告
warnings.filterwarnings("ignore", category=DeprecationWarning)


def except_hook(exc_type, exc_value, exc_tb):
    traceback.print_exception(exc_type, exc_value, exc_tb)


sys.excepthook = except_hook


class VideoPlayer(QWidget):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("视频播放器")
        self.resize(1024, 768)

        # 存储当前播放的文件路径，便于调试
        self.current_file_path = ""

        # 优先设置插件路径（针对不同平台）
        self.set_plugin_path()

        self.init_ui()
        self.init_player()

    def set_plugin_path(self):
        """设置Qt多媒体插件路径，使用用户提供的实际路径"""
        # 重点：使用你提供的实际插件路径
        user_plugin_path = r"D:\Python\Python311\Lib\site-packages\PyQt5\Qt5\plugins"

        # 额外搜索可能的解码器路径
        plugin_paths = [
            user_plugin_path,  # 优先使用用户确认的路径
            os.path.join(user_plugin_path, "multimedia"),  # 多媒体子插件目录
            os.path.join(user_plugin_path, "mediaservice"),  # 媒体服务插件目录
        ]

        # 平台特定补充路径
        if sys.platform.startswith('win'):
            plugin_paths.extend([
                # LAV Filters可能的安装路径（解码器）
                os.path.join(os.environ.get('ProgramFiles', ''), 'LAV Filters'),
                # K-Lite解码器路径
                os.path.join(os.environ.get('ProgramFiles', ''), 'K-Lite Codec Pack', 'filters'),
                # 系统默认的Qt插件路径
                os.path.join(os.environ.get('ProgramFiles', ''), 'Qt', 'plugins')
            ])

        # 添加有效的路径并去重
        added_paths = []
        for path in plugin_paths:
            # 确保路径存在且未被添加过
            if path and os.path.exists(path) and path not in added_paths:
                QApplication.addLibraryPath(path)
                added_paths.append(path)

        # 调试信息：打印已添加的插件路径（确认用户提供的路径是否被正确添加）
        print("已添加的插件路径:")
        for path in added_paths:
            print(f"  - {path}")

        # 验证多媒体插件是否存在
        multimedia_plugin_path = os.path.join(user_plugin_path, "multimedia")
        if not os.path.exists(multimedia_plugin_path):
            print(f"警告：未找到多媒体插件目录 {multimedia_plugin_path}")
            print("可能需要重新安装PyQt5以获取完整组件")

    def init_ui(self):
        # 创建视频播放区域
        self.video_frame = QFrame()
        self.video_frame.setStyleSheet("border: 2px solid #ccc;")

        self.video_widget = QVideoWidget(self.video_frame)
        video_layout = QVBoxLayout(self.video_frame)
        video_layout.addWidget(self.video_widget)
        video_layout.setContentsMargins(0, 0, 0, 0)

        # 控制按钮
        self.open_btn = QPushButton("打开文件")
        self.play_btn = QPushButton("播放")
        self.stop_btn = QPushButton("停止")

        # 进度条
        self.position_slider = QSlider(Qt.Horizontal)
        self.position_slider.setRange(0, 0)

        # 音量控制
        self.volume_slider = QSlider(Qt.Horizontal)
        self.volume_slider.setRange(0, 100)
        self.volume_slider.setValue(50)
        self.volume_label = QLabel("音量")

        # 时间显示
        self.time_label = QLabel("00:00 / 00:00")

        # 文件信息显示
        self.file_info_label = QLabel("未选择文件")
        self.file_info_label.setStyleSheet("color: #666; font-size: 12px;")

        # 控制布局
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.open_btn)
        control_layout.addWidget(self.play_btn)
        control_layout.addWidget(self.stop_btn)
        control_layout.addWidget(self.volume_label)
        control_layout.addWidget(self.volume_slider)
        control_layout.addWidget(self.time_label)

        # 主布局
        main_layout = QVBoxLayout(self)
        main_layout.addWidget(self.video_frame)
        main_layout.addWidget(self.position_slider)
        main_layout.addLayout(control_layout)
        main_layout.addWidget(self.file_info_label)

        # 绑定信号槽
        self.open_btn.clicked.connect(self.open_file)
        self.play_btn.clicked.connect(self.toggle_play)
        self.stop_btn.clicked.connect(self.stop_play)
        self.position_slider.sliderMoved.connect(self.set_position)
        self.volume_slider.valueChanged.connect(self.set_volume)

    def init_player(self):
        self.player = QMediaPlayer(None, QMediaPlayer.VideoSurface)
        self.player.setVideoOutput(self.video_widget)

        # 连接播放器信号
        self.player.stateChanged.connect(self.state_changed)
        self.player.positionChanged.connect(self.update_position)
        self.player.durationChanged.connect(self.update_duration)
        self.player.error.connect(self.handle_error)
        self.player.mediaStatusChanged.connect(self.on_media_status_changed)

    def open_file(self):
        file_path, _ = QFileDialog.getOpenFileName(
            self, "选择视频文件", "",
            "视频文件 (*.mp4 *.avi *.mov *.mkv *.flv);;所有文件 (*)"
        )

        if file_path:
            # 转换为绝对路径，避免相对路径问题
            file_path = os.path.abspath(file_path)

            # 详细的文件验证
            file_info = QFileInfo(file_path)
            validation_msg = []
            if not file_info.exists():
                validation_msg.append("文件不存在")
            if not file_info.isFile():
                validation_msg.append("不是有效的文件")
            if not file_info.isReadable():
                validation_msg.append("没有读取权限")
            if validation_msg:
                QMessageBox.warning(self, "文件验证失败", "\n".join(validation_msg))
                return

            # 显示文件信息
            self.current_file_path = file_path
            self.file_info_label.setText(f"当前文件: {os.path.basename(file_path)}")
            print(f"尝试播放文件: {file_path}")
            print(f"文件大小: {file_info.size()} 字节")

            # 设置媒体内容并播放
            self.player.setMedia(QMediaContent(QUrl.fromLocalFile(file_path)))
            self.play_btn.setText("播放")
            self.player.play()

    def on_media_status_changed(self, status):
        """处理媒体状态变化"""
        status_map = {
            QMediaPlayer.NoMedia: "无媒体内容",
            QMediaPlayer.LoadingMedia: "正在加载媒体...",
            QMediaPlayer.LoadedMedia: "媒体加载成功",
            QMediaPlayer.BufferingMedia: "正在缓冲...",
            QMediaPlayer.BufferedMedia: "缓冲完成",
            QMediaPlayer.EndOfMedia: "播放结束",
            QMediaPlayer.InvalidMedia: "无效的媒体文件"
        }
        print(f"媒体状态: {status_map.get(status, f'未知状态 ({status})')}")

    def toggle_play(self):
        if self.player.state() == QMediaPlayer.PlayingState:
            self.player.pause()
        else:
            self.player.play()

    def stop_play(self):
        self.player.stop()

    def set_position(self, position):
        self.player.setPosition(position)

    def set_volume(self, volume):
        self.player.setVolume(volume)

    def state_changed(self, state):
        if state == QMediaPlayer.PlayingState:
            self.play_btn.setText("暂停")
        else:
            self.play_btn.setText("播放")

    def update_position(self, position):
        self.position_slider.blockSignals(True)
        self.position_slider.setValue(position)
        self.position_slider.blockSignals(False)

        # 更新时间显示
        current_time = self.format_time(position)
        total_time = self.format_time(self.player.duration())
        self.time_label.setText(f"{current_time} / {total_time}")

    def update_duration(self, duration):
        self.position_slider.setRange(0, duration)

    def handle_error(self):
        error_msg = self.player.errorString()
        error_code = self.player.error()

        # 错误代码解释
        error_codes = {
            QMediaPlayer.NoError: "无错误",
            QMediaPlayer.ResourceError: "资源错误（文件不存在或无法访问）",
            QMediaPlayer.FormatError: "格式错误（不支持的文件格式）",
            QMediaPlayer.NetworkError: "网络错误",
            QMediaPlayer.AccessDeniedError: "访问被拒绝",
            QMediaPlayer.ServiceMissingError: "服务缺失（解码器问题）"
        }

        error_details = error_codes.get(error_code, f"未知错误 ({error_code})")
        QMessageBox.critical(self, "播放错误",
                             f"无法播放视频:\n{error_msg}\n\n错误类型: {error_details}")
        print(f"错误代码: {error_code} ({error_details})，信息: {error_msg}")

    def format_time(self, ms):
        """将毫秒转换为分:秒格式"""
        total_seconds = ms // 1000
        minutes = total_seconds // 60
        seconds = total_seconds % 60
        return f"{minutes:02d}:{seconds:02d}"


if __name__ == '__main__':
    app = QApplication(sys.argv)
    # 强制使用软件渲染（解决部分显卡兼容性问题）
    app.setAttribute(Qt.AA_UseSoftwareOpenGL)
    player = VideoPlayer()
    player.show()
    sys.exit(app.exec_())
