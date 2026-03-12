from PyQt5.QtWidgets import (QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLineEdit, QProgressBar, QLabel,
                             QFileDialog, QMessageBox)
from PyQt5.QtCore import QThread, pyqtSignal
from PIL import Image, ImageDraw, ImageFont
import os


# 定义水印处理线程类（直接放在主文件中）
class WatermarkThread(QThread):
    # 自定义信号：进度信号（当前进度，总任务数），完成信号（成功数，失败数）
    progress_signal = pyqtSignal(int, int)
    finish_signal = pyqtSignal(int, int)

    def __init__(self, file_list, watermark_text):
        super().__init__()
        self.file_list = file_list  # 待处理文件列表
        self.watermark_text = watermark_text  # 水印文本
        self._is_running = True  # 线程运行标志（用于优雅终止）

    def stop(self):
        # 外部调用此方法终止线程
        self._is_running = False

    def run(self):
        """线程执行入口：批量添加水印"""
        success_count = 0
        fail_count = 0
        total = len(self.file_list)

        for index, img_path in enumerate(self.file_list):
            # 检查是否需要终止线程
            if not self._is_running:
                break

            try:
                # 核心逻辑：添加水印
                self._add_watermark(img_path)
                success_count += 1
            except Exception as e:
                print(f"处理失败 {img_path}：{str(e)}")
                fail_count += 1

            # 发送进度信号（index+1：从1开始计数）
            self.progress_signal.emit(index + 1, total)

        # 任务结束，发送完成信号
        self.finish_signal.emit(success_count, fail_count)

    def _add_watermark(self, img_path):
        """添加水印的核心方法"""
        # 1. 打开图片（支持多种格式）
        with Image.open(img_path) as img:
            # 2. 准备绘制工具
            draw = ImageDraw.Draw(img)
            # 3. 设置字体（使用系统可用字体，避免找不到字体的问题）
            try:
                # Windows系统可用黑体
                font = ImageFont.truetype("C:/Windows/Fonts/simhei.ttf", 36)
            except IOError:
                try:
                    # Linux系统可用默认字体
                    font = ImageFont.truetype("/usr/share/fonts/truetype/wqy/wqy-microhei.ttc", 36)
                except IOError:
                    # 最终fallback：使用默认字体
                    font = ImageFont.load_default(size=36)
            # 4. 计算水印位置（右下角，距边缘20px）
            bbox = draw.textbbox((0, 0), self.watermark_text, font=font)
            text_width = bbox[2] - bbox[0]
            text_height = bbox[3] - bbox[1]
            img_width, img_height = img.size
            x = img_width - text_width - 20
            y = img_height - text_height - 20
            # 5. 绘制水印（半透明白色）
            draw.text(
                (x, y),
                self.watermark_text,
                font=font,
                fill=(255, 255, 255, 128)  # RGBA，最后一位是透明度
            )
            # 6. 生成保存路径（原路径添加"_watermark"后缀）
            dir_name, file_name = os.path.split(img_path)
            name, ext = os.path.splitext(file_name)
            save_path = os.path.join(dir_name, f"{name}_watermark{ext}")
            # 7. 保存图片（保持原格式）
            img.save(save_path)


# 主窗口类
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("图片批量水印工具")
        self.setGeometry(100, 100, 600, 200)

        # 1. 初始化UI控件
        self.file_label = QLabel("已选择：0个文件")
        self.select_btn = QPushButton("选择图片")
        self.watermark_input = QLineEdit()
        self.watermark_input.setPlaceholderText("请输入水印文本")
        self.start_btn = QPushButton("开始处理")
        self.progress_bar = QProgressBar()
        self.status_label = QLabel("状态：就绪")

        # 2. 布局管理
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        layout = QVBoxLayout(central_widget)

        # 文件选择行
        file_layout = QHBoxLayout()
        file_layout.addWidget(self.select_btn)
        file_layout.addWidget(self.file_label)
        layout.addLayout(file_layout)

        # 水印设置行
        watermark_layout = QHBoxLayout()
        watermark_layout.addWidget(QLabel("水印文本："))
        watermark_layout.addWidget(self.watermark_input)
        layout.addLayout(watermark_layout)

        # 控制与进度行
        control_layout = QHBoxLayout()
        control_layout.addWidget(self.start_btn)
        control_layout.addWidget(self.status_label)
        layout.addLayout(control_layout)
        layout.addWidget(self.progress_bar)

        # 3. 绑定信号槽
        self.select_btn.clicked.connect(self.select_files)
        self.start_btn.clicked.connect(self.start_process)

        # 4. 初始化变量
        self.file_list = []
        self.worker_thread = None

    # 选择图片文件
    def select_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "选择图片", "", "Image Files (*.jpg *.jpeg *.png *.bmp)"
        )
        if files:
            self.file_list = files
            self.file_label.setText(f"已选择：{len(files)}个文件")

    # 启动处理线程
    def start_process(self):
        # 校验参数
        if not self.file_list:
            QMessageBox.warning(self, "警告", "请先选择图片文件！")
            return
        watermark_text = self.watermark_input.text().strip()
        if not watermark_text:
            QMessageBox.warning(self, "警告", "请输入水印文本！")
            return

        # 避免重复启动线程
        if self.worker_thread and self.worker_thread.isRunning():
            QMessageBox.warning(self, "警告", "任务正在进行中！")
            return

        # 初始化线程
        self.worker_thread = WatermarkThread(
            file_list=self.file_list,
            watermark_text=watermark_text
        )
        # 连接线程信号
        self.worker_thread.progress_signal.connect(self.update_progress)
        self.worker_thread.finish_signal.connect(self.handle_finish)

        # 更新UI状态
        self.start_btn.setEnabled(False)
        self.status_label.setText("状态：处理中...")
        self.progress_bar.setValue(0)

        # 启动线程
        self.worker_thread.start()

    # 更新进度
    def update_progress(self, current, total):
        progress = int((current / total) * 100)
        self.progress_bar.setValue(progress)
        self.status_label.setText(f"状态：处理中 {current}/{total}")

    # 处理完成
    def handle_finish(self, success, fail):
        self.status_label.setText(f"状态：完成！成功{success}个，失败{fail}个")
        self.start_btn.setEnabled(True)
        QMessageBox.information(
            self, "处理完成",
            f"批量水印处理结束！\n成功：{success}个\n失败：{fail}个"
        )

    # 窗口关闭时清理线程
    def closeEvent(self, event):
        if self.worker_thread and self.worker_thread.isRunning():
            self.worker_thread.stop()
            self.worker_thread.wait()  # 等待线程终止
        event.accept()


if __name__ == "__main__":
    import sys
    from PyQt5.QtWidgets import QApplication

    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    sys.exit(app.exec_())
