'''
创建:
class pandas.DataFrame(data=None, index=None, columns=None,
dtype=None, copy=None)

data : 数据源，可以是以下几种形式：
列表，其中每个元素是一行数据。
字典，其中键是列名，值是列值（列表或数组）。
2d-Ndarray。
Series 对象，每个 Series 成为一列。
index : 行标签，如果没有指定，默认是整数索引 [0, ..., n-1] ，其中 n 是
数据中的行数。
columns : 列标签，如果没有指定，则列标签从数据源中推断。
dtype : 指定某列的数据类型。如果指定，则所有列都将转换为指定的数据类型。
copy : 布尔值，默认为 None 。如果为True ，则复制数据；如果为False ，则尽
可能避免复制数据。
'''

# 1.使用列表创建
# import pandas as pd
#
# data_list = ['小明','小红','小李']
#
# columns = ['姓名']
#
# df = pd.DataFrame(data_list,columns=columns,index=['a','b','c'])
#
# print(df)


# 2.嵌套列表
# import pandas as pd
# # 创建一个包含学生信息的嵌套列表，每个子列表代表一个学生的姓名、年龄和成绩
# data_list = [
#     ['小明', 20, 85],
#     ['小红', 18, 90],
#     ['小刚', 22, 88]
# ]
# # 定义列名，分别对应姓名、年龄和成绩
# columns = ['姓名', '年龄', '成绩']
# # 使用pandas库创建一个DataFrame，将数据列表和列名作为参数传入
# df = pd.DataFrame(data_list, columns=columns)
# # 打印DataFrame以查看数据
# print(df)


# 使用字典创建
# 可以使用一个字典来创建DataFrame，其中字典的键将作为列名，字典的值可以是列
# 表、数组等可迭代对象，它们的长度要一致，代表每一列的数据


# import pandas as pd
# # 定义一个字典，其中包含两组数据：姓名和年龄
# data = {
#     'Name': ['Tom', 'Nick', 'John'], # 'Name' 键对应一个包含姓名的列
#     'Age': [20, 21, 19]  # 'Age' 键对应一个包含年龄的列
# }
# # 使用pd.DataFrame()函数将字典转换为DataFrame对象
# # 这里，data字典中的键自动成为DataFrame的列名，值成为列的数据
# df = pd.DataFrame(data)
# # 打印DataFrame对象，查看其内容
# print(df)




# 使用Ndarray数组创建
# import pandas as pd
# import numpy as np
# # 定义一个二维Ndarray数组，其中包含两组数据：姓名和年龄
# data_array = np.array([
#      ['Tom', 20],
#      ['Nick', 21],
#      ['John', 19]
# ])
# # 使用pd.DataFrame()函数将二维数组转换为DataFrame对象
# df = pd.DataFrame(data_array, columns=['Name', 'Age'])
# # 打印
# print(df)





# 使用Series创建
# 如果有多个Series对象，也可以将它们组合成一个DataFrame
# import pandas as pd
# # 创建三个pandas Series对象
# s1 = pd.Series(['小明', '小红', '小刚'], name='姓名')
# s2 = pd.Series([20, 18, 22], name='年龄')
# s3 = pd.Series([85, 90, 88], name='成绩')
# # 将Series对象组合成一个字典，键是Series的名称，值是Series本身
# # 然后将这个字典传递给DataFrame构造函数来创建一个DataFrame
# df = pd.DataFrame({s1.name: s1, s2.name: s2, s3.name: s3})
# # 打印DataFrame对象，查看其内容
# print(df)


#使用concat拼接函数
# import pandas as pd
# # 创建三个pandas Series对象
# s1 = pd.Series(['小明', '小红', '小刚'], name='姓名')
# s2 = pd.Series([20, 18, 22, 0], name='年龄')
# s3 = pd.Series([85, 90, 88], name='成绩')
# # 使用concat拼接，并指定轴为1
# df = pd.concat([s1, s2, s3], axis=1)
# # 打印DataFrame对象，查看其内容
# print(df)