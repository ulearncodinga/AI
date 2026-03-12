# 第一种方式：使用 import 模块 直接导入
# import moduleA
#
#
# # 调用模块A的方式为：  moduleA.add()
# ret = moduleA.add(4, 5)
# print(ret)


# 第二种方式：使用 import 模块 as 别名
# import numpy as np
# import moduleA as mA
# ret = mA.add(2, 5)
# print(ret)



# 第三种方式：使用 from 模块 import 你想要使用的内容
# 除了导入内容之外，该模块的其他内容是不会被导入进来的
# from moduleA import add, a
# ret = add(5, 6)
# print(ret)
# import builtins



# dir()：可以用来查看模块的内置变量
# print(dir())


# import moduleA
# ret = moduleA.add(1, 3)
# print(ret)

from moduleA import *
ret = add(1, 2)
print(ret)
ret1 = sub(1, 2)
print(ret1)
# ret2 = div(1, 2)