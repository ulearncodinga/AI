#创建目录
#语法:=>
#os.mkdir(path)
#path:要创建的目录的路径
#mode:设置创建目录的权限位,默认值是0o777(八进制表示),意味着所有人都有读,
# 写和执行操作,只针对Linux系统,Windows系统会忽略
'''
如果目录创建成功,则函数不会返回任何内容.如果是指定的路径以及存在就会抛出异常,
如果路径是无效的,或者由于权限不足等原因无法创建目录,也会抛出异常

注意事项:
os.mkdir 只能创建一级目录,如果父目录不存在,则会抛出异常
如果需要创建多级目录,可以使用  os.makedirs函数, 它会递归地创建所需的中间目录
'''
# import os
# os.mkdir('./test1')#第一次会直接创建成功在左边项目栏能看到test1,再次创建会报错,因为已经存在
# # 多个目录用makedirs()




#删除目录
"""
语法=>
    import os
    os.rmdir(path)
"""
#path(字符串)  -- 要删除的空目录的路径
#如果目录删除成功,则函数不会返回任何内容,如果指定的路径不存在,则会抛出异常,
# 如果路径不是一个空目录,或者由于权限不足等原因无法删除目录,也会抛出异常

"""
注意事项:os.rmdir只能删除空目录.如果目录中包含文件或者其他目录,os.rmdir将无法删除它,并且会抛出异常
"""




#获取当前的工作目录
"""
语法=>
    os.getcwd()
该函数没有参数,但会返回一个字符串,表示当前工作目录的路径

注意事项:
    os.getcwd()返回的是字符串形式的路径
    在不同的操作系统中,路径的表示方式可能不同.例如,在Windows中路径通过使用反斜杠\,
    而在Linux系统中使用正斜杠/
适用场景:
    当需要在脚本中确定当前的工作位置时
    在进行文件操作之前,需要知道文件的相对路径时基于哪一个目录
    当需要在不同的目录之间切换时,但之后想要回到原始目录
"""
# import os
# print(os.getcwd())
#
# print(os.getcwd())
# res = open('./test1/1.txt','r+')
# res.write('1151515')


#改变当前工作目录
"""
os.chdir(path)函数来改变工作目录
import os 
os.chdir(path)

path(字符串)--要切换到的目录的路径.如果指定的路径不存在,指定的路径不是一个目录,
没有权限更改到指定的目录就会抛出异常
如果目录切换成功,则函数不会返回任何内容

"""

#列出目录下的所有内容
"""
import os
items = os.listdir(path)

path(字符串) -- path(字符串) -要列出内容的目录的路径。如果省略，默认为当前工作目录。
如果指定的路径不存在、指定的路径不是一个目录、没有权限读取指定的目录就会抛出异常
这个函数返回一个列表,其中包含指定路径下的所有文件和子目录的名称

注意事项:
    os.listdir 不会递归的列出子目录中的内容,它只列出直接位于指定目录下的文件和子目录
    返回的列表中只包含名称,不包含路径.如果需要完整路径,你需要将目录名称与路径结合起来
    如果目录为空,返回的列表将是空的
    在使用os.listdir时,应该考虑到可能出现的异常,并且适当的处理他们,以确保代码的健壮性
"""
# import os
#
# ls = os.listdir('./')
# print(ls)
# for i in ls:#使用for循环把每个文件名称遍历出来
#     print(i)
#

#重命名目录
"""
os.rename(sec,dst)函数对一个文件或者目录进行重命名,这个函数可以将一个文件或目录从其当前的
路径(源路径)更改到一个新的路径(目标路径)
    src(字符串): 要重命名的文件或目录的当前路径
    dst(字符串):文件或者目录的新名称和路径
如果命名成功,则函数不会返回任何内容
"""
# import os
# src = './t1/t3/666.txt'
# dst = './t1/t3/66.txt'
# os.rename(src,dst)

#检查路径是不是目录
"""
os.path.isdir()函数检查给定路径是否为目录
"""
# import os
# print(os.path.isdir('./1.txt'))
# print(os.path.isdir('./t1'))



#检查路径是不是文件
"""
os.path.isfile()函数检查给定路径是否为文件
语法:
        import os
        is_file = os.path.isfile(path)
是则返回True
不是则返回False

用途:
1.
在处理文件之前，确认指定的路径确实是一个文件，这样可以避免对目录执行文件操作。
2.
在脚本中执行条件逻辑时，根据路径是否为文件来决定下一步操作。
3.
在文件处理脚本中，区分文件和目录，以便进行适当的操作。
"""
# import os
# print(os.path.isfile('./目录的操作.py'))





#路径拼接
"""
将一个或者多个路径进行组合合并成一个完整的路径.这个函数会根据操作系统的文件系统约定来正确的处理路径分隔符
语法:=>
    import os
    os.path.join(path,*paths)

path(字符串):起始路径,通常是一个目录路径
*paths(可变参数):需要连接到path的其他路径的片段
返回一个字符串,表示将所有的
"""
import os

path = './'

ls = os.listdir(path)

new_ls = []
print(ls)
for name in ls:
    res = os.path.join(path,name)
    if res[-2:] == 'py':
        # with open(res,'r+') as fd:
        #     fd.write('abc')#把这些文件都添加abc
        print(res)


#路径拆分
"""
os.path.split(path)函数将路径分割成两部分:目录路径和文件名
语法:=>
    import os
    head,tail = os.path.split(path)
path:要分割的路径
head:path的目录路径
tail:path的文件名

主要用途:
1.
    从完整路径中提取文件名或目录名。
2.
    用于文件处理时，需要单独操作文件名和路径的其他部分。
3.
    在遍历文件系统时，帮助确定每个文件的上级目录
"""

import os
path = './1.txt'
p,name = os.path.split(path)
print(p,name)


#获取绝对路径
import os
path = './1.txt'
res = os.path.abspath(path)
print(res)



# #检查路径是否存在
import os
res = os.path.exists('./open函数.py')
print(res)

# import os.path
# is_dir = os.path.isfile('./目录的操作.py')
# print(is_dir)