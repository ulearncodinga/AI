# #第一种方式,使用import 模块直接导入
# import moduleA
# #调用模块A的方式为: moduleA.add()
# ret = moduleA.add(4,5)
# print(ret)

#第二种方式:使用import 模块 as 别名
# import moduleA as ma
#
# ret = ma.add(4, 5)
# print(ret)

#第三种方式:使用from 模块 import 你想要的内容
#除了导入内容之外,其他内容不会被导入
# from moduleA import add
# ret = add(5,6)
# print(ret)

# import builtin

#dir()可以查看模块的内置变量
#print(dir())

# import moduleA
# ret = moduleA.add(1,3)
# print(ret)

from moduleA import *

ret = add(1, 2)
ret2 = sub(1,2)
# ret3 = mul(4,5)#这个mul被__all__限制住了所以调用这个会报错

print(ret)
print(ret2)
# print(ret3)
