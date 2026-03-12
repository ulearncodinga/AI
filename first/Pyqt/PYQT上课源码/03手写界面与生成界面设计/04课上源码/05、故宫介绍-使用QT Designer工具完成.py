"""
案例：
    使用外部工具设计界面，实现功能
"""

# 这条调用是需要自己写的，.ui生成的文件中只有一个类，不能直接运行
from my_window_ui import Ui_Form

import sys
from PyQt5.QtWidgets import QApplication, QWidget

class MyWindow(QWidget, Ui_Form):
    def __init__(self):
        """
        构造函数
        """
        super().__init__()
        # 调用函数，添加界面需要的组件
        self.setupUi(self)
        # 设计标题
        self.setWindowTitle("使用Designer工具设计的界面")
        # 初始化槽函数
        self.slot_init()

    def slot_init(self):
        self.show_btn.clicked.connect(self.show_info)

    def show_info(self):
        """
        显示故宫介绍
        :return:
        """
        self.show_txt.setText("故宫建筑群以中轴线为中心，布局严谨有序，体现了中国古代建筑的艺术风格和哲学思想。"
                              "屋顶上铺着黄色的琉璃瓦，在阳光下闪闪发光，显得气势磅礴；红色的墙面上，雕刻着各种图案，细节精致。"
                              "故宫的建筑风格典雅瑰丽，展现了中国古代建筑的高超技艺和独特魅力。")


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = MyWindow()
    window.show()
    app.exec()










