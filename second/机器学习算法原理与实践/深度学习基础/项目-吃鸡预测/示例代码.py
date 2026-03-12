import os
import random
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import torch
from sklearn.preprocessing import StandardScaler
from torch.utils.data import TensorDataset,DataLoader
from torch import nn,optim


def setup_seed(seed):
    np.random.seed(seed)
    random.seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    torch.manual_seed(seed)#创建cuda随机数种子
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True

setup_seed(0)

if torch.cuda.is_available()