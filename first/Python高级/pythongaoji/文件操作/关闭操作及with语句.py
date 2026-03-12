"""
close()方法关闭文件,他释放了与文件对象关联的系统资源,并且确保数据正确地写入存储介质
功能:
释放资源 :关闭文件，释放与文件对象关联的所有系统资源，如文件描述符、缓冲区等。
刷新缓冲区: 在关闭文件之前，它会自动刷新文件的内部缓冲区，确保所有缓冲的数据都被写入磁盘。
进制进一步操作 关闭文件后，文件对象不再允许进行读取、写入或其他操作。
"""
"""
with语句:
with语句是一种上下文管理器
语法=>
with expression [as variable]:
    with-block
执行流程:
执行表达式
调用__enter__()方法
执行语句块
调用__exit__()方法
"""
from io import TextIOWrapper
path = '.1/txt'
with open('test1/1.txt', 'r+') as fd:
    res =fd.read()
    print(res)