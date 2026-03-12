"""
read(size):size是可选参数,文本模式下,一次最多读取文件指针后面size个大小的字符,在二进制模式下,一次最多读取文件指针后面size个大小的字节
默认值为None,表示一次性读取文件指针后面的所有的内容并将其作为字符串返回
readline():从文件里读取单行数据
readlines():读取所以行,返回一个列表

"""
path = 'test1/1.txt'
res = open('test1/1.txt', 'r')
# read_str = res.read(14)
# print(read_str)
#
# path = './1.txt'
# res = open('./1.txt','r+',encoding='utf-8')
# read_str = res.read(14)
# print(read_str)

# read_str1 = res.readline()
# print(read_str1)
# read_str2 = res.readlines()
# print(read_str2)

"""
写入文件
write(str):


"""

res = open('test1/1.txt', 'r+', encoding='utf-8')
num = res.write('aaaa')
print(num)#点开1.txt文件可以看到前四个以及变为aaaa
res.close()