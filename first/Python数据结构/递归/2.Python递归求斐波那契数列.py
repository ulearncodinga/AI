'''使用Python递归函数计算斐波那契数列。
斐波那契数列，又称黄金分割数列，其中每个数字是前两个数字
的总和。这个数列以0和1开始，之后的斐波那契项可以通过将前
两项相加得到。
F(0)=0
F(1)=1
F(n)=F(n−1)+F(n−2)'''

#循环方式
# def fib(n):
#     '''生成斐波那契数列值的第n个数字'''
#     fib_list = []
#     a,b = 0,1
#     while len(fib_list) < n:
#         fib_list.append(a)
#         a,b = b,a+b
#     return fib_list
#
# print(f"循环方式:{fib(10)}")




#递归方式实现斐波那契数列
def fibonacci(n):
    if n <= 0:
        return []
    elif n == 1:
        return [0]
    elif n == 2:
        return [0,1]
    else:
        fibonacci_list = fibonacci(n - 1)
        fibonacci_list.append(fibonacci_list[-1] + fibonacci_list[-2])  #返回一个列表,列表包含(0,n-1)个斐波那契数
        return fibonacci_list


print(fibonacci(10))






















