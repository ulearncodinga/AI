''''''
''''
                图片输入
                          }  ->与运算->图像融合->图像输出
模板输入->灰度化->二值化
'''
import cv2
import numpy as np
import matplotlib.pyplot as plt

# 设置图像显示的中文字体（避免中文标签乱码）
plt.rcParams['font.sans-serif'] = ['SimHei']
plt.rcParams['axes.unicode_minus'] = False


# 读取目标图像和水印图像
def read_images(target_path, watermark_path):
    # 读取目标图像（默认读取彩色图像，保留BGR通道）
    target_img = cv2.imread('lena.png')
    if target_img is None:
        raise ValueError("目标图像读取失败，请检查文件路径是否正确")

    # 读取水印图像（保留Alpha通道，用于后续透明背景处理）
    watermark_img = cv2.imread('logo.png', cv2.IMREAD_UNCHANGED)
    if watermark_img is None:
        raise ValueError("水印图像读取失败，请检查文件路径是否正确")

    return target_img, watermark_img


# 调整水印图像尺寸（确保不超过目标图像的1/4，避免遮挡主体）
def resize_watermark(target_img, watermark_img, scale=0.25):
    # 获取目标图像和水印图像的尺寸
    target_height, target_width = target_img.shape[:2]
    watermark_height, watermark_width = watermark_img.shape[:2]

    # 计算水印图像的最大允许尺寸（目标图像的scale比例）
    max_watermark_width = int(target_width * scale)
    max_watermark_height = int(target_height * scale)

    # 保持水印图像的宽高比，调整尺寸
    scale_ratio = min(max_watermark_width / watermark_width, max_watermark_height / watermark_height)
    new_watermark_size = (int(watermark_width * scale_ratio), int(watermark_height * scale_ratio))
    resized_watermark = cv2.resize(watermark_img, new_watermark_size, interpolation=cv2.INTER_AREA)

    return resized_watermark


# 主函数中调用
if __name__ == "__main__":
    # 图像文件路径（请根据实际情况修改）
    target_path = "target_image.jpg"
    watermark_path = "watermark.png"

    # 读取图像
    target_img, watermark_img = read_images(target_path, watermark_path)

    # 调整水印尺寸
    resized_watermark = resize_watermark(target_img, watermark_img, scale=0.2)

    # 显示原始图像和调整后的水印图像
    fig, axes = plt.subplots(1, 2, figsize=(15, 6))

    # OpenCV读取的图像为BGR通道，Matplotlib显示需要转换为RGB通道
    axes[0].imshow(cv2.cvtColor(target_img, cv2.COLOR_BGR2RGB))
    axes[0].set_title("目标原始图像")
    axes[0].axis("off")

    # 处理水印图像的通道显示（如果有Alpha通道，只显示RGB部分）
    if resized_watermark.shape[-1] == 4:
        axes[1].imshow(cv2.cvtColor(resized_watermark[:, :, :3], cv2.COLOR_BGR2RGB))
    else:
        axes[1].imshow(cv2.cvtColor(resized_watermark, cv2.COLOR_BGR2RGB))
    axes[1].set_title("调整尺寸后的水印图像")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()