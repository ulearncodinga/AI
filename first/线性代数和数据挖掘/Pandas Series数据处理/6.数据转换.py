'''
1.replace 替换特定的值,使用字典映射替换

Series.replace(to_replace=None, value=None, inplace=False, limit=None,
regex=False, method='pad')

to_replace ：要替换的值，可以是以下类型：
标量：单个值。
列表：一系列值。
字典：键是要替换的值，值是替换后的新值。
正则表达式：如果 regex=True ，则可以使用正则表达式匹配要替换的值。
value ：替换后的新值，可以是标量或字典。如果 to_replace 是列表，则 value 也应该是相同
长度的列表。
inplace ：布尔值，表示是否在原地修改数据。如果为 True ，则直接在原 Series 上修改，不
返回新的对象。
limit ：整数，表示最大替换量。如果指定，则只替换前 limit 个匹配的值。
regex ：布尔值，表示是否将 to_replace 解释为正则表达式。
method ：字符串，表示填充的方法，在 to_replace 参数是一个标量、列表或元组，同时 value
参数设置为 None 时，可以使用 method 参数来指定填充缺失值（NaN）的方式。可选值包括：
pad / ffill ：用前一个非缺失值去填充缺失值。
bfill ：用后一个非缺失值去填充缺失值。

'''

import pandas as pd
# 创建一个 Series
s = pd.Series([1, 2, 3, 4, 5])
# 进行替换操作
replaced = s.replace(to_replace=2, value=20)
print(replaced)


'''
astype()：用于将 Series 的数据类型（dtype）转换或转换为另一种类型。

Series.astype(dtype, copy=True, errors='raise')

dtype：你希望将 Series 转换成的数据类型。
copy：布尔值，默认为 True 。如果为 False ，则转换数据类型时不会复制底层数据（如果可能
的话）。
errors：默认为 ‘raise’，控制当转换失败时的行为。如果设置为 ‘raise’，则在转换失败时会抛出异
常；如果设置为 ‘ignore’，转换失败后则返回原始 Series ，不做任何修改。
'''


# import pandas as pd
# s = pd.Series([1, 2, 3, 4, 5])
# # 将整数转换为字符串
# s_str = s.astype(float)
# print(s_str)


# transform()：用于对Series中的数据进行转换操作，并返回与原始Series具有相同索引的新
# Series。


# Series.transform(func, axis=0, *args, **kwargs)
'''
func : 应用于Series的函数。这个函数可以是内置函数，或者自定义的函数。
axis : 对于Series来说，这个参数不起作用，因为Series是一维的。在DataFrame上使用时，
axis=0 （默认）表示按列应用函数， axis=1 表示按行应用函数。
*args, **kwargs : 这些参数会被传递给 func 函数。
'''
import pandas as pd
# 自定义一个函数，返回每个元素的平方
def square(x):
    return x ** 2
# 创建一个Series
s = pd.Series([1, 2, 3, 4, 5])
# 使用自定义函数对Series进行平方变换
transformed_series = s.transform(square)
print(transformed_series)




