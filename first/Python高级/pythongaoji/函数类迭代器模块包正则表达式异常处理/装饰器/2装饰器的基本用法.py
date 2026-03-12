# #这个函数只能打印a
# # def func():
# #     print('a')
# # func()
# #现在用装饰器给他添加功能
#
# def decorator(f):
#     def f1():
#         print('b')
#         f()
#         print('c')
#     return f1
# def func():
#     print('a')
#
# func = decorator(func)#传入要装饰的函数名,不能带括号,带括号就是调用函数了
# func()
# """
# 其中，func()函数原本的功能是打印字符串a，但是将其作为参数传给func1并将结果赋值给func之后，再次执行
# func的结果就是先打印字符串b，再打印字符串a，然后打印字符串c。
#
# 在这个过程中，原函数func的代码并没有被修改，而最后执行func()的结果确是和之前大不相同了，原因就在于func作为了参数传递给了decorator
# ，并且在decorator函数中添加了新的内容，且最后的返回值又重新赋值给了func，也就是说，此时的func函数以及不是原先的打印字符串
# a的函数了，而是decorator函数所返回的f1函数，那么下面的func()实际上等同于f1()。那么这段代码本质上是修改了调用的函数，
# 但是并没有修改调用方式，并且还实现了附加的功能。
# 如果将func作为参数传递给decorator而返回值并没有重新赋值给func的话，那么它就是一个闭包函数。
# 但是如果返回值重新赋值给func的话,那么decorator函数就是一个装饰器函数，它将对接收到的func参数添加打印字符串b和c的功能，且不影响func原有的功能。
# """
#
#
# #装饰器语法糖
# #   作用
# #       让代码简洁
# #语法糖:
# #    列表推导式 , 集合推导式
# #    字典推导式,  链式比较 1<a<3(用的比较少)
# #    f-string,  解包
# #     装饰器
# """
# 在Python中，装饰器的使用可通过在目标函数上方放置一个@符号，随后跟上装饰器函数的名字来实现。
# 这样，我们就无需显式地将目标函数作为参数传递给装饰器函数，而是可以直接调用目标函数。
# 通过这种方式，装饰器能够为原有函数增添额外的功能。比如上面的例子可以改为：

# def decorator(f):
#     def f1():
#         print('b')
#         f()
#         print('c')
#     return f1
# @decorator#相当于func = decorator(func)
# def func():
#     print('a')
# func()
#
#
#
#
#
#
# def decorator(f):#接收一个函数
#     def f1():
#         print('b')
#         f()
#         print('c')
#     return f1#返回一个新的函数
#
#
#
# def func():#这是原函数,我要用装饰器往里面添加内容
#     print('a')
#
#
# func = decorator(func)
# func()


















