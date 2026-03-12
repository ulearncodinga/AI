"""
装饰器本身带参数,而装饰器本身又要接收被装饰的函数名称,并且返回一个新的函数,
为了保证装饰器能够顺利接收被装饰的函数,需在装饰器
内部多一层内嵌函数,最外层的函数用来获取装饰器的参数,
中间的的函数用来获取被装饰的函数名称,最内层的函数负责进行功能的添加
"""
# def func1(prefix):
#     def decorator(func):
#         def wrapper(*args,**kwargs):
#             result = func(*args,**kwargs)
#             return f"{result}{prefix}"
#         return wrapper
#     return decorator
# #使用装饰器并传递参数
# @func1("How are you?")
# def greet(name):
#     return f"Hello,{name}."
# # greet = func1('How are you?')(greet)
# #调用被装饰的函数
# print(greet('Jack'))


import time
def g(x):
    def outer(func):
        def w(*args,**kwargs):
            res = func(*args,**kwargs)
            if x == 'file':
                print("文件账号密码验证")
            elif x == 'mysql':
                print("mysql账号密码验证")
            return res
        return w
    return outer
@g("file")
def func1():
    print("func1")
func1()












