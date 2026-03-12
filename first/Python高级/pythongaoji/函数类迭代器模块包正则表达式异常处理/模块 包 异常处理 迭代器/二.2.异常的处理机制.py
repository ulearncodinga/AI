"""
在Python中，
使用try、except、finally、else等关键字来组合成不同的异常处理方式，
其中try和except是异常处理机制中的核心。
异常处理分为三种，
分别为捕获 单个异常 、 捕获多个异常 和 捕获全部异常 ，
在捕获到对应的异常后，不会再导致程序终止执行，而是会执行处理异常的代码。

"""

#单个异常处理
"""
格式：
try：
    # 有可能发生异常的代码
except 某个异常：
    # 异常发生后要执行的代码
else：
    # 如果没有异常发生，在try执行完毕后执行这里的代码
finally：
    # 不管有没有捕获到异常，最后都会执行这里的代码
"""
#捕获多个异常处理
"""
方法一
格式：
try：
    # 有可能发生异常的代码
except (第一个异常，第二个异常)：
    # 异常发生后要执行的代码
else：
    # 如果没有异常发生，在try执行完毕后执行这里的代码
finally：
    # 不管有没有捕获到异常，最后都会执行这里的代码

方法二
格式:
捕获多个异常
格式：
try：
# 有可能发生异常的代码
except 第一个异常：
    # 异常发生后要执行的代码
except 第二个异常:
    #异常发生后要执行的代码
...(except可以写多个)

else：
    # 如果没有异常发生，在try执行完毕后执行这里的代码
finally：
    # 不管有没有捕获到异常，最后都会执行这里的代码
"""

#捕获所有异常
"""
格式：
try：
    # 有可能发生异常的代码
except Exception：
    # 异常发生后要执行的代码
else：
    # 如果没有异常发生，在try执行完毕后执行这里的代码
finally：
    # 不管有没有捕获到异常，最后都会执行这里的代码
"""
#使用
# 1. KeyboardInterrupt:用户主动结束程序去触发
# import time
# for i in range(100):
#     print(i)
#     time.sleep(1)

# 2.AttributeError:尝试访问对象所没有的属性时触发
# my_str = 'abcdef'
# my_str.abc()
#
# # 3.IndexError:序列中没有索引时触发
# my_list = [1,2,3,4]
# print(my_list[10])
#
# # 捕获单个异常
try:
    x = 10
    y = 0
    print(x/y)
except ZeroDivisionError:
    print('除数不能为0')
else:
    print("代码正常运行,没有出错")
finally:
    print("不管有没有出错,我都会运行")


#捕获多个异常
try:
    print(a)
    x =10
    y = 1
    z =x /y
except ZeroDivisionError:
    print("除数不为0")
except NameError:
    print("有变量没有定义")
else:
    print(z)


#捕获所有异常
try:
    print(a)
    x =0
    y = 0
    z = x / y
except Exception as e:
    print(e)

#自定义异常
# import time
# for i in range(100):
#     print(i)
#     time.sleep(1)
#     raise NameError('手动抛出异常')