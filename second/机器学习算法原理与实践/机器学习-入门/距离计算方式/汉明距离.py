import math

x = [1,2]
y = [4,6]

def hamming_distance(x_str,y_str):
    return sum(a!= b for a,b in zip(x_str,y_str))

x_str = '101100'
y_str = '111000'

print("汉明距离:",hamming_distance(x_str,y_str))