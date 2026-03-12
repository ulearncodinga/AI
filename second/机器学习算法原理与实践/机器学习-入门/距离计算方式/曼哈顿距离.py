''''''
'''
曼哈顿距离主要用于离散空间类型的距离计算

'''
import math

x = [1,2]
y = [4,6]
def manhattan_distance(x,y):
    return sum([abs(a-b) for a,b in zip(x,y)])

print("曼哈顿距离:",manhattan_distance(x,y))
