# Python递归实现斐波那契数列

# 循环方式
def fib(n):
    """生成斐波那契数列，直到第n个数字"""
    fib_list = []
    a, b = 0, 1
    while len(fib_list) < n:
        fib_list.append(a)
        a, b = b, a + b
    return fib_list

print(f'循环方式：{fib(10)}')

# 递归方式
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0, 1]
    else:
        fibonacci_list = fibonacci(n - 1) # 达到 n == 1
        fibonacci_list.append(fibonacci_list[-1] + fibonacci_list[-2]) # 返回一个列表，这个列表包含 0个到 n-1个斐波那契数
        return fibonacci_list

print(f'递归的方式：{fibonacci(2)}')








