import pandas as pd
import numpy as np
import random
import os

import torch


#设置随机种子保证结果的可重复性
def setup_seed(seed):
    #设置numpy随机数种子,确保numpy生成的随机数序列一致
    np.random.seed(seed)

    #设置python内的随机数种子,保证Python内置的随机函数生成的随机数一致
    random.seed(seed)

    #设置Python哈希种子,避免不同运行环境夏哈希结果不同,影响随机数生成
    os.environ['PYTHONHASHSEED'] = str(seed)

    #设置Pytorch随机种子,使Pytorch生辰大哥随机数序列可以重复
    torch.manual_seed(seed)

    #检查是否有可用的CUDA设备(GPU)
    if torch.cuda.is_available():
        #设置CUDA随机种子,保证在GPU上随即操作可以重复
        torch.cuda.manual_seed(seed)
        #为所有GPU设置随机种子
        torch.cuda.manual_seed_all(seed)
        #关闭cudnn自动寻找最优算法加速的功能,保证结果可重复
        torch.backends.cudnn.benchmark = False
        #设置cudnn为确定性算法,确保每次运行结果一致
        torch.backends.cudnn.deterministic = True
setup_seed(0)