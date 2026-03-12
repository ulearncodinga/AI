


# import pandas as pd
#
# data = {
#     '姓名': ['小明', '小红', '小刚'],
#     '年龄': [20, 18, 22],
#     '成绩': [85, 90, 88]
# }
#
# df = pd.DataFrame(data, index=['a', 'b', 'c'])
# print(df)
# print(df['姓名'])
# print(df[['姓名', '年龄']])
# print(df['成绩']['b'])


#
# import pandas as pd
#
# data = {
#     '姓名': ['小明', '小红', '小刚'],
#     '年龄': [20, 18, 22],
#     '成绩': [85, 90, 88]
# }
#
# df = pd.DataFrame(data, index=['a', 'b', 'c'])
# print(df)
# print(df.loc['a', '姓名'])
# print(df.iloc[1, 0])
# print(df.loc['a':'b', '姓名':'成绩'])
# # print(df.iloc[0, 0])
# print(df.iloc[0:1, 0:1])


#
# import pandas as pd
#
# data = {
#     '姓名': ['小明', '小红', '小刚'],
#     '年龄': [20, 18, 22],
#     '成绩': [85, 90, 88]
# }
#
# df = pd.DataFrame(data)
# print(df)
# print(df.at[0, '姓名':'年龄'])
# print(df.iat[0, 0])

#
#
# import pandas as pd
#
# # 创建一个示例DataFrame
# data = {
#     'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# }
# df = pd.DataFrame(data)
#
# # 使用head方法获取前5行
# print(df.head(3))


#
# import pandas as pd
#
# # 创建一个示例DataFrame
# data = {
#     'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
#     'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
# }
# df = pd.DataFrame(data)
#
# # 使用tail方法获取最后5行
# print(df.tail())


# import pandas as pd
#
# # 创建一个示例DataFrame
# data = {
#     'A': [1, 2, 3, 4, 5],
#     'B': ['a', 'b', 'c', 'd', 'e']
# }
# df = pd.DataFrame(data)
#
# print(df)
#
# # 检查DataFrame中的元素是否包含在指定的值集合中
# values_to_check = [2, 4, 'c']
# print(df.isin(values_to_check))
#
# print(df['C'])



import pandas as pd

data = {
    '姓名': ['小明', '小红', '小刚'],
    '年龄': [20, 18, 22],
    '成绩': [85, 90, 88]
}

df = pd.DataFrame(data)

# 获取'成绩'列
scores = df.get('成绩')

print(scores)

# 尝试获取不存在的列，返回指定值
non_existent_column = df.get('体重', default='Not Found')


print(non_existent_column)