#获取光标位置
#一个中文三个字节
# 在python中使用tell()函数去返回当前文件的
import time
from io import TextIOWrapper
path = 'test1/1.txt'
with open('test1/1.txt', 'a+') as fd:
    res = fd.tell()
    print(res)

#改变文件指针的位置
"""seek()函数
语法=>
seek(offset,whence=0)
offset :
whence : 默认为0,指定offset的起始位置
0:表示从文件开头计算偏移量(默认值)
1:表示当前文件指针位置开始计算偏移量
2:表示从文件末尾开始计算偏移量
注意事项:
seek
方法在文本模式和二进制模式下都有效，但文本模式whence只能使用默认值，不能自己修改。
seek方法基于字节偏移量。这意味着即使文件包含多字节字符，seek改变的仍然是字节偏移量。
"""
# path = './1.txt'
# with open('1.txt','r+') as fd:
#     res = fd.tell()
#     print(res)
#     fd.seek(5)
#     ret = fd.read()
#     print(ret)


"""
获取文件属性
获取文件大小:os.path.getsize()
"""
#获取文件大小
import os
import time

print(os.path.getsize('./open函数.py'))#打印main文件占用的大小(单位是字节)

#获取文件的最后修改时间
last_time = os.path.getmtime('./open函数.py')
format = "%Y-%m-%d  %H:%M:%S"
lotime = time.localtime(last_time)
res = time.strftime(format,lotime)
print(res)

#获取文件的创建时间
# 语法=>
# os.path.getctime()

# 获取文件最后访问时间
# 语法=>
#         os.path.getatime(path)