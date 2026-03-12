"""
闭包函数的:如果内函数使用了外函数的局部变量,并且外函数把内函数返回出来的过程叫闭包
"""
#闭包函数具体实例
# def nth_power(exponent):
#     def exponent_of(base):
#         return base ** exponent
#     return exponent_of
#
# square = nth_power(2)
# cube = nth_power(3)
#
# print(square(3))
# print(cube(3))

"""
在Python中，装饰器（Decorator）本质上是一种特殊的嵌套函数，它接收一个函数作为参数（该函数就是被装饰的函数），
并返回一个新的函数（装饰之后的函数）。装饰器最大的作用就是可以让我们在不改变被装饰函数的代码的情况下去给它添加新的功能
--------
装饰器的概念(特殊的嵌套函数)接收一个函数作为参数(这个函数就是被装饰的函数)并返回一个新的函数(装饰后的函数)
作用:
    让我们在不改变被装饰函数的代码的情况下给它添加新的功能
"""
# import time
#
# def outer(func):
#     def wrapper(*args,**kwargs):
#         res = func(*args,**kwargs)
#         return res
#     return wrapper
# def fun1():
#     start = time.time()
#     time.sleep(2)
#     print("func1")
#     end = time.time()-start
#     print(end)
# def fun2(x):
#     time.sleep(2)
#     print(f"func2 {x}")
#
# wrapper = outer(fun2)
# fun1()

'''
给fun1加入权限认证功能,需要用户正确的账号密码之后才能调用fun1否则不能
正确的账号和密码分别是 gscsd 123456
# '''
def outer(func):
    def wrapper(*args,**kwargs):
        res = func(*args,**kwargs)
        while True:
            zhanghao1 = input("请输入你的账号:")
            password1 = input("请输入你的密码:")
            if zhanghao1 == 'gscsd' and password1 == '123456':
                print("登陆成功")
                break
            else:
                print("登陆失败,请重新输入")
        return res
    return wrapper
# @outer
# def fun1():
#         pass
# fun1()
def fun1():
    pass
wrapper = outer(fun1)
wrapper()







def out(f):
    def d(*args,**kwargs):
        res = f(*args,**kwargs)
        while True:
            z = input("请输入你的账号:")
            p = input("请输入你的密码")
            if z == 'aaa' and p == '123':
                print("登陆成功")
                break
            else:
                print("登陆失败")
        return res
    return d
@out
def fun1():
    pass
fun1()
# d = out(fun1)
# d()















