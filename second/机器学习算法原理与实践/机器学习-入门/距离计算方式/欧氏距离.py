import math
#新建两个点
x = [1,2]
y = [4,6]

def eucalidean_siatance(x,y):
    return math.sqrt(sum([(a-b) ** 2 for a,b in zip(x,y)]))

print("欧氏距离:",eucalidean_siatance(x,y))