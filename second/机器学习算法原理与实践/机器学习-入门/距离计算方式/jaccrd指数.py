import math

x = [1,2]
y = [4,6]

def jaccrd_index(x_set,y_set):
    intersection = len(set(x_set&y_set))
    union = len(set(x_set | y_set))
    return intersection / union
x_set = {1,2,3}
y_set = {2,3,4}

print("jaccrd指数:",jaccrd_index(x_set,y_set))