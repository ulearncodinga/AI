

# # 1.index
# import pandas as pd
#
# data = {
#     '姓名': ['小明', '小红', '小刚'],
#     '年龄': [20, 18, 22],
#     '成绩': [85, 90, 88]
# }
#
# df = pd.DataFrame(data, index=[3, 4, 5])
# print(df)
#
# print(df.index)
# print(df.columns)
# print(df.values)
# print(df.dtypes)
# print(df.shape)
# print(df.size)

#
# # empty
# import pandas as pd
# import numpy as np
#
# data = [None]
#
# df = pd.DataFrame(data, index=['a'])
# print(df)
#
# print(df.empty)


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
#
# res = df.T
# print(res)


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
#
# print(df.axes)



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