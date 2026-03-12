# # 创建一个列表，这是一个内置的可迭代对象
my_list = [1, 2, 3, 4, 5]
#
# # 使用 iter() 函数获取列表的迭代器
list_iterator = my_list.__iter__()
#
print(list_iterator)
print(list_iterator.__next__())
print(list_iterator.__next__())
print(list_iterator.__next__())
#
#
# # new_list = reversed(my_list)
# # print(new_list)
# # print(type(list_iterator))
# # print(list_iterator)
#
# # 使用迭代器的 __next__() 方法来遍历列表
# try:
#     while True:
#         # 获取下一个元素
#         item = list_iterator.__next__()
#         print(item)
# except StopIteration:
#     # 当没有更多元素时，迭代器会抛出 StopIteration 异常
#     print("Iteration is complete.")
#
#
# for i in my_list:
#     print(i)



#
# # 创建一个字典，这是一个内置的可迭代对象
# my_dict = {1: 'one', 2: 'two', 3: 'three'}
#
# # 使用 iter() 函数获取字典的迭代器
# dict_iter = my_dict.__iter__()
# # print(type(dict_iter))
#
#
# # 使用迭代器的 __next__() 方法来遍历字典
# try:
#     while True:
#         # 获取下一个元素
#         item = dict_iter.__next__()
#         print(item)
# except StopIteration:
#     # 当没有更多元素时，迭代器会抛出 StopIteration 异常
#     print('Iteration is complete')








#
# class MyRangeIterator:  # 定义一个名为MyRangeIterator的类
#     def __init__(self, end):  # 构造函数，初始化迭代器
#         self.end = end  # 设置迭代器的结束值
#         self.num = 0  # 设置当前迭代的数值，初始为0
#
#     def __iter__(self):  # 实现__iter__方法，返回迭代器本身
#         return self  # 因为我们的迭代器就是实例本身，所以返回self
#
#     def __next__(self):  # 实现__next__方法，返回下一个值
#         if self.num < self.end:  # 如果当前数值小于结束值
#             value = self.num  # 获取当前数值作为要返回的值
#             self.num += 1  # 将当前数值加1，为下一次迭代做准备
#             return value  # 返回当前数值
#         else:  # 如果当前数值已经达到或超过结束值
#             raise StopIteration  # 抛出StopIteration异常，结束迭代
#
# # 使用MyRangeIterator类创建一个迭代器实例，迭代0到4的整数
# for i in MyRangeIterator(5):  # for循环会自动调用迭代器的__next__方法
#     print(i)  # 打印迭代器返回的每个值







# class FibonacciIterator:  # 定义一个名为FibonacciIterator的类
#     def __init__(self, n):  # 构造函数，初始化迭代器
#         self.n = n  # 设置迭代器的结束值
#         self.a, self.b = 0, 1  # 初始化斐波那契数列的前两个数
#         self.count = 0  # 初始化计数器，用于跟踪迭代次数
#
#     def __iter__(self):  # 实现__iter__方法，返回迭代器本身
#         return self  # 因为我们的迭代器就是实例本身，所以返回self
#
#     def __next__(self):  # 实现__next__方法，返回下一个值
#         if self.count < self.n:  # 如果当前计数器小于结束值
#             value = self.a  # 获取当前斐波那契数作为要返回的值
#             self.a, self.b = self.b, self.a + self.b  # 更新斐波那契数列的下一个数
#             self.count += 1  # 增加计数器
#             return value  # 返回当前斐波那契数
#         else:  # 如果当前计数器已经达到或超过结束值
#             raise StopIteration  # 抛出StopIteration异常，结束迭代
#
# # 使用FibonacciIterator类创建一个迭代器实例，迭代生成前10个斐波那契数
# fib_iterator = FibonacciIterator(10)  # 创建一个生成前10个斐波那契数的迭代器
#
# for number in fib_iterator:  # for循环会自动调用迭代器的__next__方法
#     print(number)  # 打印迭代器返回的每个值
#
#
#
