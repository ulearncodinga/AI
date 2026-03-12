# 如果a + b + c = 1000，并且a ^ 2 + b ^ 2 = c ^ 2(a，b，c为自然数)，使用Python求出a、b、c可能的组合。
import time

start_time = time.time()
for a in range(0, 1001):
    for b in range(0, 1001):
        for c in range(0,1001):
            if a + b + c == 1000 and a ** 2 + b ** 2 == c ** 2:
                print(f'a:{a}, b:{b}, c:{c}')
end_time = time.time()
print(f'程序运行总时间为：{end_time - start_time}')

start_time = time.time()
for a in range(0, 1001):
    for b in range(0, 1001):
        c = 1000 - a - b
        if a ** 2 + b ** 2 == c ** 2:
            print(f'a:{a}, b:{b}, c:{c}')
end_time = time.time()
print(f'第二次程序运行总时间为：{end_time - start_time}')

