'''
拉格朗日表示法 f'(x)
莱布尼茨表示法 df(x)/dt
'''
'斜率的极值和极小值'
import numpy as np

# 设置随机种子（确保结果可重复）
np.random.seed(42)

# 1. 生成2行3列[0,1)随机浮点数
arr1 = np.random.rand(2, 3)
print("rand数组：\n", arr1)

# 2. 生成3行3列[1,10)随机整数
arr2 = np.random.randint(1, 10, (3, 3))
print("randint数组：\n", arr2)

# 3. 打乱1维数组
arr3 = np.array([1,2,3,4,5])
np.random.shuffle(arr3)
print("打乱后：", arr3)  # 输出：[4 5 3 1 2]
