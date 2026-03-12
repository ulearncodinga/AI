import tkinter as tk
import random
import threading
import time

# 定义温馨提示语列表
tips = [

]

# 定义背景颜色列表（更多柔和色彩选择）
bg_colors = [
    '#FFCCD5', '#C5E3ED', '#D4F1F4', '#E9E4F0',
    '#FFF8E6', '#FFE8D6', '#F1ECC3', '#D6EFD8',
    '#FFE6E6', '#E8F4F8', '#FFF0F3', '#F5F3FF',
    '#FFF5E6', '#E6F7F0', '#FFEAEA', '#F0F7FF'
]

# 定义字体颜色列表（与背景色形成良好对比）
font_colors = [
    '#D81F26', '#00529B', '#137333', '#5C2D91',
    '#C8553D', '#E67E22', '#718096', '#4A5568'
]


def show_warm_tip():
    # 随机选择提示语、背景色和字体颜色
    tip = random.choice(tips)
    bg_color = random.choice(bg_colors)
    font_color = random.choice(font_colors)

    # 创建临时窗口获取屏幕信息
    temp = tk.Tk()
    screen_width = temp.winfo_screenwidth()
    screen_height = temp.winfo_screenheight()
    temp.destroy()

    # 需求一：固定窗口大小（统一为300x150）
    win_width = 300
    win_height = 150

    # 需求二：全屏均匀随机分布算法
    # 1. 确保窗口不会超出屏幕范围
    max_x = screen_width - win_width
    max_y = screen_height - win_height

    # 2. 完全随机分布，覆盖整个屏幕
    x = random.randint(0, max_x)
    y = random.randint(0, max_y)

    # 创建弹窗
    window = tk.Toplevel()
    window.title("生日祝福")
    window.geometry(f"{win_width}x{win_height}+{x}+{y}")
    window.attributes("-topmost", True)  # 窗口置顶
    window.configure(bg=bg_color)

    # 需求三：美观字体设置
    # 使用更优雅的字体和合适的大小，添加加粗效果
    tk.Label(
        window,
        text=tip,
        bg=bg_color,
        fg=font_color,  # 字体颜色
        font=("Microsoft YaHei UI", 14, "bold"),  # 美观字体设置
        wraplength=win_width - 40,  # 自动换行
        padx=20,
        pady=20
    ).pack(expand=True)

    # 添加关闭按钮
    tk.Button(
        window,
        text="谢谢",
        command=window.destroy,
        font=("Microsoft YaHei UI", 10),
        bg="#FFD700",
        fg="#333333",
        padx=10,
        pady=2
    ).pack(pady=10)

    window.mainloop()


if __name__ == "__main__":
    # 控制弹窗数量和弹出间隔
    total_windows = 50000  # 弹窗总数
    for i in range(total_windows):
        t = threading.Thread(target=show_warm_tip)
        t.daemon = True  # 主线程结束后自动退出
        t.start()
        # 随机间隔时间，避免过于规律
        time.sleep(random.uniform(0.005, 0.010))