# 1.index 行标签
#   columns列标签
#   values返回的数据用Ndarray表示

# import pandas as pd
# data ={
#     '姓名': ['小明','小红','小李'],
#     '年龄':[18,19,20],
#     '成绩':[81,82,83]
# }
# df = pd.DataFrame(data,index=['a','b','c'])
# print(df)
#
# print(df.index)
# print(df.columns)
# print(df.values)



# 在Datavalues中只能用dtypes
# dtypes
# 返回每列的数据类型



'''
shape:是个二维的
返回DataFrame的形状（行数，列数）。
'''
'''
size
返回DataFrame中的元素数量。
'''
'''
empty  有NAN和None都不算为空
返回DataFrame是否为空。
'''
'''
T
返回DataFrame的转置。
'''

'''
axes
返回行轴和列轴的列表。
'''

'''
ndim
返回DataFrame的维度数。对于标准的二维DataFrame，这个值通常是2。
'''

'''attrs
允许用户存储DataFrame的元数据，它是一个字典，可以用来存储任意与
DataFrame相关的额外信息'''


import pandas as pd
data = {
 '姓名': ['小明', '小红', '小刚'],
 '年龄': [20, 18, 22],
 '成绩': [85, 90, 88]
}
df = pd.DataFrame(data, index=['a', 'b', 'c'])
print(df)
df.attrs['creator'] = 'xxxxx'
df.attrs['created_at'] = '2024-13-32'
print(df.attrs)