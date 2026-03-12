import torch
# 检查是否支持 CUDA
print("CUDA 可用:", torch.cuda.is_available())
# 查看 CUDA 版本（需与安装时的 cu128 对应）
print("CUDA 版本:", torch.version.cuda)
# 查看当前使用的显卡（若有）
if torch.cuda.is_available():
    print("当前显卡:", torch.cuda.get_device_name(0))