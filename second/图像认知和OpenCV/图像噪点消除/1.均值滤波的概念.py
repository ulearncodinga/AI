# 噪点:指图像中的干扰因素
# 常见的噪声类型:高斯噪声(符合正态分布)和 椒盐噪声(随机出现黑点白点)
''''''
'''
均值滤波(线性,快)
    边缘模糊化最明显
方框滤波(线性)

高斯滤波(线性)
    对高斯噪声去除效果最好
    实时应用
    

中值滤波(非线性,慢)
    对椒盐噪声去除效果最好
双边滤波(非线性,最慢)
    对边缘保留效果最好
    核值不断变化
    过滤严重噪声的离线应用
    
'''
'''

均值滤波:

原始图像   滤波器(卷积核):构建权重分布
(将滤波器在原始图像中滑动计算,决定周围需要补几圈,
 将权重分别乘以范围内的像素值然后再相加,得到中心点的像素值)
'''

import cv2
import numpy as np
import matplotlib.pyplot as plt  # 用于合并显示多子图


def load_image(image_path):
    """读取图像，返回BGR格式的图像数组（OpenCV默认）"""
    image = cv2.imread('lena.png')
    if image is None:
        raise ValueError("无法读取图像，请检查路径是否正确或图像格式是否支持（优先png/jpg）")
    return image


def mean_filter(image, ksize=3):
    """均值滤波：邻域像素取平均，简单快速但易模糊细节"""
    return cv2.blur(image, (ksize, ksize))


def box_filter(image, ksize=3, ddepth=-1, normalize=True):
    """方框滤波：可切换归一化（等价均值滤波）/非归一化（像素求和）"""
    return cv2.boxFilter(image, ddepth, (ksize, ksize), normalize=normalize)


def gaussian_filter(image, ksize=3, sigmaX=1.5):
    """高斯滤波：中心像素权重高，兼顾降噪与细节保留，适合高斯噪声"""
    return cv2.GaussianBlur(image, (ksize, ksize), sigmaX)


def median_filter(image, ksize=3):
    """中值滤波：邻域像素排序取中值，擅长消除椒盐噪声，无边缘模糊"""
    return cv2.medianBlur(image, ksize)


def bilateral_filter(image, d=5, sigmaColor=50, sigmaSpace=50):
    """双边滤波：结合空域+值域权重，降噪同时精准保留边缘细节"""
    return cv2.bilateralFilter(image, d, sigmaColor, sigmaSpace)


def merge_and_show_results(original, mean_img, box_img, gaussian_img, median_img, bilateral_img,
                           save_path="./results/"):
    """
    合并显示所有结果（单图多子图）+ 保存单张结果图
    参数：
        original: 原始图像（BGR格式）
        mean_img~bilateral_img: 5种滤波结果（BGR格式）
        save_path: 结果保存目录
    """
    # 1. 预处理：创建目录 + BGR转RGB（matplotlib默认RGB显示）
    import os
    if not os.path.exists(save_path):
        os.makedirs(save_path)  # 不存在则创建结果目录

    # OpenCV读取的是BGR格式，matplotlib需转为RGB格式才能正常显示颜色
    def bgr2rgb(img):
        return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # 转换所有图像格式
    original_rgb = bgr2rgb(original)
    mean_rgb = bgr2rgb(mean_img)
    box_rgb = bgr2rgb(box_img)
    gaussian_rgb = bgr2rgb(gaussian_img)
    median_rgb = bgr2rgb(median_img)
    bilateral_rgb = bgr2rgb(bilateral_img)

    # 2. 合并显示：2行3列布局（共6个子图）
    plt.figure(figsize=(18, 12))  # 设置整体图大小（宽18英寸，高12英寸），可根据屏幕调整
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'SimHei']  # 支持英文/中文标签
    plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示异常

    # 子图1：原始图像
    plt.subplot(2, 3, 1)  # 第2行、第3列、第1个子图
    plt.imshow(original_rgb)
    plt.title("Original Image\n（原始图像）", fontsize=14, fontweight='bold')
    plt.axis('off')  # 隐藏坐标轴，避免干扰视觉

    # 子图2：均值滤波
    plt.subplot(2, 3, 2)
    plt.imshow(mean_rgb)
    plt.title("Mean Filter\n（均值滤波：简单快速，细节模糊）", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 子图3：方框滤波（归一化=均值滤波）
    plt.subplot(2, 3, 3)
    plt.imshow(box_rgb)
    plt.title("Box Filter (Normalized)\n（方框滤波：等价均值，可切换求和模式）", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 子图4：高斯滤波
    plt.subplot(2, 3, 4)
    plt.imshow(gaussian_rgb)
    plt.title("Gaussian Filter\n（高斯滤波：高斯噪声优选，细节保留好）", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 子图5：中值滤波
    plt.subplot(2, 3, 5)
    plt.imshow(median_rgb)
    plt.title("Median Filter\n（中值滤波：椒盐噪声克星，无边缘模糊）", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 子图6：双边滤波
    plt.subplot(2, 3, 6)
    plt.imshow(bilateral_rgb)
    plt.title("Bilateral Filter\n（双边滤波：降噪+边缘保留，效果最佳）", fontsize=14, fontweight='bold')
    plt.axis('off')

    # 调整子图间距，避免标签重叠
    plt.tight_layout(pad=3.0)  # pad=3.0：子图间间距（英寸），可按需调整

    # 3. 保存结果：同时保存“合并图”和“单张结果图”
    # 保存合并图（高清，dpi=300）
    merge_save_path = os.path.join(save_path, "all_filters_comparison.png")
    plt.savefig(merge_save_path, dpi=300, bbox_inches='tight')  # bbox_inches='tight'：去除多余空白
    print(f"合并对比图已保存至：{merge_save_path}")

    # 保存单张结果图（保持原始分辨率）
    single_saves = [
        (original, "original.jpg", "原始图像"),
        (mean_img, "mean_filter.jpg", "均值滤波结果"),
        (box_img, "box_filter.jpg", "方框滤波结果"),
        (gaussian_img, "gaussian_filter.jpg", "高斯滤波结果"),
        (median_img, "median_filter.jpg", "中值滤波结果"),
        (bilateral_img, "bilateral_filter.jpg", "双边滤波结果")
    ]
    for img, filename, desc in single_saves:
        save_path_single = os.path.join(save_path, filename)
        cv2.imwrite(save_path_single, img)
        print(f"{desc}已保存至：{save_path_single}")

    # 显示合并图（弹窗形式，关闭弹窗后程序继续）
    plt.show()


if __name__ == "__main__":
    # -------------------------- 配置参数（可按需修改） --------------------------
    image_path = "./demo.png"  # 测试图像路径（请确保该路径下有demo.png文件）
    ksize = 3  # 卷积核大小（建议奇数：3/5/7，越大降噪越强但模糊越明显）
    sigmaX = 1.5  # 高斯滤波标准差（越大模糊越强，1.0~2.0为常用范围）
    d = 5  # 双边滤波邻域直径（实时场景建议5，离线高噪声场景可设9）
    sigmaColor = 50  # 双边滤波颜色相似度权重（50~100，越大颜色混合范围越广）
    sigmaSpace = 50  # 双边滤波空间距离权重（50~100，越大空间影响范围越广）
    save_root = "./filter_results/"  # 结果保存根目录（自动创建）
    # ---------------------------------------------------------------------------

    # 1. 读取原始图像
    try:
        original_image = load_image(image_path)
        print(f"成功读取原始图像，分辨率：{original_image.shape[1]}x{original_image.shape[0]}（宽x高）")
    except ValueError as e:
        print(f"图像读取失败：{e}")
        exit()  # 读取失败则退出程序

    # 2. 执行5种滤波算法
    mean_result = mean_filter(original_image, ksize=ksize)
    box_result = box_filter(original_image, ksize=ksize, normalize=True)  # 归一化=均值滤波
    gaussian_result = gaussian_filter(original_image, ksize=ksize, sigmaX=sigmaX)
    median_result = median_filter(original_image, ksize=ksize)
    bilateral_result = bilateral_filter(original_image, d=d, sigmaColor=sigmaColor, sigmaSpace=sigmaSpace)
    print("所有滤波算法执行完成，开始合并显示与保存...")

    # 3. 合并显示+保存结果
    merge_and_show_results(
        original=original_image,
        mean_img=mean_result,
        box_img=box_result,
        gaussian_img=gaussian_result,
        median_img=median_result,
        bilateral_img=bilateral_result,
        save_path=save_root
    )
    print("所有操作完成！")
