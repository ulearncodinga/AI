'''
1.1 概念
NumPy(Numerical Python)是一个开源的Python科学计算库，旨在为Python提供
高性能的多维数组对象和一系列工具。NumPy数组是Python数据分析的基础，许多
其他的数据处理库（如Pandas、SciPy）都依赖于NumPy。以下是NumPy的一些主
要特点：
1. 高性能：NumPy底层主要采用C语言编写，相比于python来说运算速度快，性能
优越，并且保留了python的易用性。
2. 多维数组：NumPy提供了强大的n维数组对象ndarray，可进行高效的数据处
理，这是Numpy进行数据处理的核心对象。
3. 丰富的函数：NumPy内置了大量数学、统计和线性代数函数，方便进行数据计
算，除此之外还具有广播功能，可以允许不同形状的数组进行算术运算。
4. 广泛的接口：NumPy与许多其他科学计算库（如Matplotlib、SciPy）兼容，可
轻松实现数据交换和集成，此外，在人工智能领域中也可以很方便的和神经网络
中使用的张量进行结构转换
'''
#安装命令: pip install numpy
# numpy可以转换为张量
#用做科学计算

'''
Ndarray与list区别
list效率低
Ndarray:
1. Ndarray数组所有元素的数据类型相同、数据地址连续，批量操作数组元素时速
度更快，而list中元素的数据类型可能不同，需要通过寻址的方式找到下一个元
素。
2. Ndarray数组支持广播机制，矩阵运算时不需要写for循环。
3. 底层主要使用C语言实现，运行速度远高于Python代码。


'''
my_list = [1,2,3,4,5]
for i in range(len(my_list)):
    my_list[i] +=1
print(my_list)

#使用Ndarray数组
import numpy as np
arr = np.array([1,2,3,4,5])
# Numpy数组可以直接进行加1操作，它会令数组中的所有的元素都去进行加1
arr = arr + 1
print(arr)

import numpy as np

a = np.array([1,2,3,4,5])
b = np.array([1,2,3,4,5])
c = a + b
print(c)






