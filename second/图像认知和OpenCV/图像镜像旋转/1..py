import cv2
import os

if __name__ == "__main__":
    # 1. 定义图像路径
    img_path = "./lena.png"
    save_path = "./lena_both_flip.png"

    # 2. 读取彩色图像
    src_img = cv2.imread(img_path)
    if src_img is None:
        print(f"错误：无法读取图像，请检查路径 '{img_path}' 是否正确！")
        exit()

    # 3. 水平垂直翻转（flipCode = -1，小于0即可，常用-1）
    both_flip_img = cv2.flip(src_img, flipCode=-1)

    # 4. 显示对比（创建拼接图，方便直观对比）
    # 拼接原始图像和翻转图像（水平拼接）
    compare_img = cv2.hconcat([src_img, both_flip_img])

    cv2.namedWindow("Original vs Both Flip", cv2.WINDOW_NORMAL)
    cv2.imshow("Original vs Both Flip", compare_img)

    # 5. 保存与退出
    key = cv2.waitKey(0)
    if key == ord("s"):
        cv2.imwrite(save_path, both_flip_img)
        cv2.imwrite("./lena_compare.png", compare_img)
        print(f"水平垂直翻转后的图像已保存至：{save_path}")
        print(f"对比图已保存至：./lena_compare.png")

    cv2.destroyAllWindows()