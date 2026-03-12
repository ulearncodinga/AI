#读写之前必须打开文件,读写之后必须关闭文件

#打开文件
"""
open函数来打开文件,这个函数返回一个文件,可以用来进行后续的读写等操作

file_name:打开文件的路径加名称
mode:打开文件模式,默认为'r',表示只读模式且文本模式读取
buffering：可选参数，缓冲区大小。0表示无缓冲，1表示行缓冲，更大的整数表示具体的缓冲区大小。默认为None，表示默认的缓冲策略，大多数情况下，使用默认值就可以了。
encoding：可选参数，用于指定文件的编码，仅适用于文本模式，默认值None表示使用系统的默认编码来打开文本文件。
errors：可选参数，用于指定如何处理编码和解码错误，对于二进制模式无效。常见的值有strict、ignore、replace等。
newline：可选参数，用于控制通用换行符模式的行为。它可以是None、''、'\n'、'\r'或'\r\n'。如果设置为None，
则通用换行符模式被启用，\n、\r和\r\n都被识别为换行符，并以\n的形式在文本模式下读取。如果设置为其他值，则在该值处进行换行符的转换。
closefd：可选参数，如果为True（默认值），则在文件关闭时关闭文件描述符。如果为False，则文件描述符在文件关闭时保持打开状态。

"""
# 格式=>
# open(file_name,mode='r',buffering=None,encoding=None,errors=None,newline=None,closefd=True)
path = 'test1/1.txt'
res = open('test1/1.txt', 'r')
print(res)