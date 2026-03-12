import cv2
import matplotlib.pyplot as plt

# 设置中文字体（避免matplotlib显示中文乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False

if __name__ == "__main__":
    # 读取图像
    path = "./lena.png"
    image_np = cv2.imread(path)
    if image_np is None:
        print("图像读取失败，请检查路径是否正确！")
        exit()

    # 缩放参数：x轴放大2倍，y轴放大2倍（便于观察放大后的细节差异）
    fx = 2.0
    fy = 2.0

    # 定义五种插值算法及其名称（用于显示）
    interpolations = [
        (cv2.INTER_NEAREST, "最近邻插值"),
        (cv2.INTER_LINEAR, "双线性插值"),
        (cv2.INTER_AREA, "像素区域插值"),
        (cv2.INTER_CUBIC, "立方插值"),
        (cv2.INTER_LANCZOS4, "Lanczos插值")
    ]

    # 创建画布，显示5种算法的结果（1行5列）
    plt.figure(figsize=(20, 4))

    for i, (interp, name) in enumerate(interpolations):
        # 图像缩放
        resize_img = cv2.resize(image_np, None, fx=fx, fy=fy, interpolation=interp)
        # 转换颜色空间：OpenCV读取的是BGR，matplotlib显示的是RGB
        resize_img_rgb = cv2.cvtColor(resize_img, cv2.COLOR_BGR2RGB)
        # 绘制图像
        plt.subplot(1, 5, i + 1)
        plt.imshow(resize_img_rgb)
        plt.title(name, fontsize=12)
        plt.axis("off")  # 隐藏坐标轴

    # 保存图像（可选）
    plt.savefig("./interpolation_comparison.png", dpi=300, bbox_inches="tight")
    # 显示画布
    plt.show()

    # 关闭所有窗口
    cv2.destroyAllWindows()