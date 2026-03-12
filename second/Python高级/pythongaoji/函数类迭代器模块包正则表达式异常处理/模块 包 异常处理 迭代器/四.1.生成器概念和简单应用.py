"""
生成器的概念
生成器（Generator）
是一种特殊的迭代器，它可以在需要时动态生成值，避免一次性生成所有值所占用的大量内存。
生成器使用yield语句来产生值，可以通过迭代器协议来逐个获取生成器产生的值。
生成器可以在循环中被逐个迭代，从而实现高效地处理大量数据或者无限序列。
在Python中，生成器可以通过函数定义和生成器表达式来创建。


生成器
的主要特点包括：
1惰性求值
生成器不会在创建时生成所有值，而是逐个生成值。这意味着生成器可以在需要的时候再去生成值。
2.
使用yield关键字：
只有yield被调用时，生成器才会返回一个值，并且程序暂停执行，直到下一次迭代时才继续执行。
3.内存效率：
生成器可以处理大型数据集或生成大型数据结构，而不会占用太多内存，因为它们只在需要时生成值。
4.
迭代器协议：
生成器也是迭代器，所有实现了迭代器协议，因此可以用于for循环和其他需要迭代器的代码中。


生成器函数的主要特点:
使用yield关键字：
普通函数使用return返回值，而生成器函数使用yield关键字返回值。
状态保持：生成器函数在每次产生一个值之后会暂停执行，并保持当前的状态，包括局部变量的状态和当前的执行位置。当再次调用生成器时，他会从上次yield语句之后的地方继续执行。
惰性求值：生成器只在需要时才计算并生成值，这意味着它可以高效处理大量数据或无限流数据。
"""





#定义生成器函数
def count_up_to(max):
    count = 1
    while count <=max:
        yield count
        count +=1
res = count_up_to(5)
print(res)


#使用生成器函数
for number in count_up_to(5):
    print(number)


# def fibonacci():
#     a,b = 0,1
#     while True:
#         yield a
#         a,b = b,a+b
# #打印前十个斐波那契数
# fib = fibonacci()
# for _ in fib:
#     print(_)


# ls = list(range(1000))
# for i in ls:
#     print(i)



# def make_num():
#     start = 1
#     while True:
#         yield start
#         start += 1
# num_gen = make_num()
# for i in num_gen:
#     print(i)

# # 执行顺序
# class A:
#     def __init__(self):
#         print('A')
#
# class B1(A):
#     def __init__(self):
#         super().__init__()  # 初始化父类参数
#         print('B1')
#
# class B2(A):
#     def __init__(self):
#         super().__init__()  # 初始化父类参数
#         print('B2')
#
# class C(B1,B2):
#     def __init__(self):
#         super().__init__()  # 初始化父类参数
#         print('C')
#
# print(C.mro())
# c = C()  # B2 B1  C
