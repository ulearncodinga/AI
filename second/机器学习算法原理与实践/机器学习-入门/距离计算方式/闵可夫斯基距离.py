''''''
'''
闵可夫斯基距离是欧式距离和曼哈顿距离的泛化。
p=1 -曼哈顿距离
p=2 -欧氏距离
p=∞- 切比雪夫距离
'''
import math

x = [1,2]
y = [4,6]

def minkovski_distance(x,y,p):
    return sum(abs(a - b) ** p for a,b in zip(x,y)) ** (1/p)
p = 2
print("闵可夫斯基距离:",minkovski_distance(x,y,p))