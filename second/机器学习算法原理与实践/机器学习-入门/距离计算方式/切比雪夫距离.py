import math

x = [1,2]
y = [4,6]

def chebyshev_distance(x,y):
    return max([abs(a-b) for a,b in zip(x,y)])
print("切比雪夫距离:",chebyshev_distance(x,y))