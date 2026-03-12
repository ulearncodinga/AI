

# # 位置索引
# import pandas as pd
#
# # 创建一个Series
# series = pd.Series([10, 20, 30, 40, 50], index=[4, 3, 2, 1, 0])
#
# # 通过位置索引获取元素
# print(series)
# print(series[0])
# print(series[1])



# # 标签索引
# import pandas as pd
#
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'a', 'c', 'd', 'e'])
#
# print(series)
# # 通过标签索引获取元素
# ser1 = series['a']
# print(type(ser1))
#
# ser2 = series['c']
# print(type(ser2))



# # 切片
# import pandas as pd
#
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
#
# print(series)
#
# # # 通过位置切片
# # print(series[:])
#
# # 通过标签切片
# print(series['d':'b'])



# # loc iloc
# import pandas as pd
#
# 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])

# print(series)

# 使用.loc通过标签索引
# print(series.loc[0])
# print(series.loc['f'])
#
# # 使用.iloc通过位置索引
# print(series.iloc[0:2])
# print(series.iloc['a'])


# # at  iat
# import pandas as pd
#
# # 创建一个带有标签的Series
# series = pd.Series([10, 20, 30, 40, 50], index=['a', 'b', 'c', 'd', 'e'])
#
# print(series)
#
# # 使用.at通过标签索引
# print(series.at['a'])
#
# # 使用.iat通过位置索引
# print(series.iat[0])
# print(series.iat[2])



# # head
# import pandas as pd
#
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50, 60, 70, 80, 90, 100]
# index = ['a', 'b', 'c', 'd', 'e', 'f', 'g', 'h', 'i', 'j']
# series = pd.Series(data, index=index)
#
# # print(series)
#
# # 使用head函数查看前5行数据
# print(series.tail(-3))



# # isin
# import pandas as pd
#
# # 创建一个Series对象
# data = [10, 20, 30, 40, 50]
# series = pd.Series(data)
#
# print(series)
#
# # 指定要判断的一组值
# values_to_check = [20, 40]
#
# # 使用isin函数进行判断
# result = series.isin(values_to_check)
#
# print(result)



# get
import pandas as pd

# 创建一个示例Series
s = pd.Series(['apple', 'banana', 'cherry'], index=[1, 2, 3])

# 使用get方法获取元素
print(s.get(2))
print(s.get(4, 'Not Found'))