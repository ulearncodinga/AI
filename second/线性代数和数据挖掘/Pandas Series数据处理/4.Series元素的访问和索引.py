'''
1.位置索引
可以使用整数索引来访问Series中的元素，就像访问列表一样
2.标签索引
除了使用位置索引之外，还可以使用标签进行索引，与访问字典中的元素类似。
3.切片索引
Series对象的切片方式有两种，第一种是使用位置切片，其使用方法与列表的切片类似；第二种是使用
标签切片，其语法与位置切片类似，都是 start:stop ，且开始值与终止值可以省略，但与位置切片不
同的是，标签切片的范围是左右都闭合，即既包含start，又包含stop，而位置切片是左闭右开，只包含
start，不包含stop。
'''

#
# # 位置索引
# import pandas as pd
# # 创建一个Series
# series = pd.Series([10, 20, 30, 40, 50])
# # 通过位置索引获取元素
# print(series[0])
# print(series[2])



# 标签索引
# import pandas as pd
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
# # 通过标签索引获取元素
# print(series['a'])
# print(series['c'])
#
# # 切片索引
# import pandas as pd
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
# # 通过位置切片(左闭右开)
# print(series[:])
# # 通过标签切片(左闭右闭 )
# print(series['b':'d'])


'''
 loc与iloc
loc与iloc也是Series对象的属性，它们的作用就是用来访问Series中的元素，
loc是基于标签的索引，
iloc是基于位置的索引。
'''
# import pandas as pd
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
# # 使用.loc通过标签索引
# print(series.loc['a'])
# # 使用.iloc通过位置索引
# print(series.iloc[0:2])
# print(series.iloc[2])

'''
at与iat
at与iat也是Series对象的属性，可以用来访问元素，at是基于标签的索引，iat是基于位置的索引
'''
# import pandas as pd
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
# # 使用.at通过标签索引
# print(series.at['a'])
# # 使用.iat通过位置索引
# print(series.iat[0])
# print(series.iat[2])


'''
head
head是Series对象的方法，用于快速查看 Series 数据的开头部分内容。

series.head(n=None)
n:指定返回的行数,默认值为5
'''
# import pandas as pd
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# series = pd.Series(data,index= index)
# print(series.head())

'''
tail
tail的用法与head类似，但不同的是，它用于快速查看 Series 数据的末尾部分内容。

series.tail(n=None)
'''
# import pandas as pd
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# series = pd.Series(data, index=index)
# # 使用tail函数查看后5行数据
# print(series.tail())


'''
isin
该函数用于判断 Series 中的每个元素是否在指定的一组值中，它会返回一个与原 Series 长度相同的布
尔型 Series ，其中对应位置为 True 表示该位置的元素在指定的值集合中， False 则表示不在。

series.isin(values)
values ：是一个可迭代对象（如列表、元组、集合等），用于指定要进行判断的一组值。
'''

# import pandas as pd
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50]
# series = pd.Series(data)
# # 指定要判断的一组值
# values_to_check = [20, 40]
# # 使用isin函数进行判断
# result = series.isin(values_to_check)
# print(result)


'''
get
Series.get 方法用于通过标签来获取Series中的元素。
Series.get(key, default=None)

key : 你想要获取的元素的标签。
default : 可选参数，如果 key 不在标签中，返回这个默认值。如果没有指定，默认为 None 。


'''
import pandas as pd
# 创建一个示例Series
s = pd.Series(['apple', 'banana', 'cherry'], index=[1, 2, 3])
# 使用get方法获取元素
print(s.get(2))
print(s.get(4, 'Not Found'))