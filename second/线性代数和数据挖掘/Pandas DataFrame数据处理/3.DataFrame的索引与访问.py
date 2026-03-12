'''
使用列名访问
对于DataFrame来说，可以直接使用列名来访问某一列的数据，返回的是一个Series
对象。

'''
import pandas as pd
data = {
 '姓名': ['小明', '小红', '小刚'],
 '年龄': [20, 18, 22],
 '成绩': [85, 90, 88]
}
df = pd.DataFrame(data)
print(df)
print(df['姓名'])
print(df[['姓名', '年龄']])



'''
loc(标签索引)和iloc(位置索引,填写标签会报错)
可以使用loc 与iloc 属性访问单个或多个数据，其语法为：

df.loc[row_label, column_label]
'''
# import pandas as pd
# data = {
# '姓名': ['小明', '小红', '小刚'],
# '年龄': [20, 18, 22],
# '成绩': [85, 90, 88]
# }
# df = pd.DataFrame(data)
# print(df)
# print(df.loc[0, '姓名'])
# print(df.loc[0:2, '姓名':'成绩'])
# print(df.iloc[0, 0])
# print(df.iloc[0:1, 0:1])


'''at和iat
使用at 和 iat 属性来访问单个数据。
'''
# import pandas as pd
# data = {
#  '姓名': ['小明', '小红', '小刚'],
#  '年龄': [20, 18, 22],
#  '成绩': [85, 90, 88]
# }
# df = pd.DataFrame(data)
# print(df)
# print(df.at[0, '姓名'])
# print(df.iat[0, 0])



'''
head
该方法用于获取DataFrame的前 n 行。默认情况下，如果不指定 n 的值，它会返回
前5行。

DataFrame.head(n=5)

'''

import pandas as pd
# 创建一个示例DataFrame
data = {
'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}
df = pd.DataFrame(data)
# 使用head方法获取前5行
print(df.head())

'''
tail
该方法用于获取DataFrame的最后 n 行。与 DataFrame.head 方法类似，如果不指
定 n 的值，它会默认返回最后5行。
'''
import pandas as pd
# 创建一个示例DataFrame
data = {
'A': [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
 'B': [10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
}
df = pd.DataFrame(data)
# 使用tail方法获取最后5行
print(df.tail())

'''
isin
DataFrame.isin(values) 方法用于检查DataFrame中的元素是否包含在指定的集
合 values 中。这个方法会返回一个布尔型DataFrame，其中的每个元素都表示原
始DataFrame中对应元素是否在 values 中。

DataFrame.isin(values)
'''
import pandas as pd
# 创建一个示例DataFrame
data = {
'A': [1, 2, 3, 4, 5],
'B': ['a', 'b', 'c', 'd', 'e']
}
df = pd.DataFrame(data)
# 检查DataFrame中的元素是否包含在指定的值集合中
values_to_check = [2, 4, 'c']
print(df.isin(values_to_check))


'''
get
用于从DataFrame中获取列，它类似于直接使用 df[key] 来访问列，但是当列不存
在时， get 方法提供了一个更安全的方式来处理这种情况，你可以指定一个默认值，
而不是引发一个错误。

DataFrame.get(key, default=None)
'''

import pandas as pd
data = {
 '姓名': ['小明', '小红', '小刚'],
 '年龄': [20, 18, 22],
 '成绩': [85, 90, 88]
}
df = pd.DataFrame(data)
# 获取'成绩'列
scores = df.get('成绩')
# 尝试获取不存在的列，返回指定值
non_existent_column = df.get('体重', default='Not Found')
print(scores)
print(non_existent_column)