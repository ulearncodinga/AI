'''
to_csv
用于将DataFrame对象保存为CSV（逗号分隔值）文件的方法。


DataFrame.to_csv(path_or_buf=None, sep=',', na_rep='', float_format=None,
columns=None, header=True, index=True, mode='w', encoding=None, quoting=None,
quotechar='"'， **kwargs)


path_or_buf : 指定输出文件的路径或文件对象。
sep : 字段分隔符，通常使用逗号 , 或制表符 \t 。
na_rep : 缺失值的表示方式，默认为空字符串 '' 。
float_format : 浮点数的格式化方式，例如 '%.2f' 用于格式化为两位小数。
columns : 要写入的列的子集，默认为None，表示写入所有列。
header : 是否写入列名，通常设置为True。
index : 是否写入行索引，通常设置为True或False，取决于是否需要索引。
mode : 写入模式，通常使用’w’（写入，覆盖原文件）或’a’（追加到文件末尾）。
encoding : 文件编码，特别是在处理非ASCII字符时很重要
quoting : 控制字段引用的行为，通常用于确保字段中的分隔符被正确处理。
quotechar : 用于包围字段的字符，默认为双引号 " 。
**kwargs : 其他关键字参数。
'''


# import pandas as pd
# # 创建一个简单的DataFrame
# data = {
#  '姓名': ['张三', '李四', '王五'],
#  '年龄': [28, 34, 29],
#  '城市': ['北京', '上海', '广州']
# }
# df = pd.DataFrame(data)
# # 将DataFrame保存为CSV文件
# df.to_csv('人员信息.csv', index=False, encoding='utf_8_sig')

"""
to_excel
在操作excel表格之前，需要安装一个库： openpyxl ，因为 pandas 库本身并不包含写入Excel文件的直
接支持，如果没有安装的话将无法操作excel。
其安装命令为：

pip install openpyxl

Pandas中的 to_excel 用于将DataFrame保存为Excel文件

DataFrame.to_excel(excel_writer, sheet_name='Sheet1', na_rep='', 
float_format=None, columns=None, header=True, index=True, index_label=None, 
startrow=0, startcol=0, engine=None, merge_cells=True, inf_rep='inf', 
freeze_panes=None, storage_options=None)

excel_writer ：字符串或ExcelWriter对象，指定输出文件的路径或文件对象。
sheet_name='Sheet1' ：要写入的工作表名称。
na_rep='' ：指定缺失值的表示方式。
float_format=None ：浮点数的格式化方式，例如’%.2f’。
columns=None ：要写入的列的子集，默认为None，表示写入所有列。
header=True ：是否写入列名，默认为True。
index=True ：是否写入行索引，默认为True。
index_label=None ：指定行索引列的列名，如果为None，并且 header 为True，则使用索引
名。
startrow=0 ：写入DataFrame的起始行位置，默认为0。
startcol=0 ：写入DataFrame的起始列位置，默认为0。
engine=None ：指定用于写入文件的引擎，可以是’openpyxl’（默认）或’xlsxwriter’。
merge_cells=True ：是否合并单元格，这在有合并单元格的Header时很有用。
inf_rep='inf' ：指定无限大的表示方式。
freeze_panes=None ：指定冻结窗口的单元格范围，例如’A2’。
storage_options=None ：指定存储连接的参数，例如认证凭据

"""

# import pandas as pd
# # 创建一个简单的DataFrame
# data = {
#    '姓名': ['张三', '李四', '王五'],
#    '年龄': [28, 34, 29],
#    '城市': ['北京', '上海', '广州']
# }
# df = pd.DataFrame(data)
# # 将DataFrame保存为Excel文件
# df.to_excel('人员信息.xlsx', index=False)

'''
read_csv
pandas.read_csv 是一个非常强大的函数，用于从文件、URL、文件-like对象等读
取逗号分隔值（CSV）文件。这个函数有很多参数，允许你以多种方式自定义数据加
载过程。

pandas.read_csv(filepath_or_buffer, sep, header, usercols, 
na_values, parse_dates, skiprows, nrows)

filepath_or_buffer ：指定要读取的 CSV 文件的路径或文件对象。可以是一个
字符串，表示文件的绝对路径或相对路径；也可以是一个已经打开的文件对象
（例如通过 open() 函数打开的文件）。
sep : 字符串，用于分隔字段的字符。默认是逗号 , ，但可以是任何字符，例如
';' 或 '\t' （制表符）。
header : 整数或整数列表，用于指定行号作为列名，或者没有列名（例如
header=None ）。默认为 'infer' ，表示自动检测列名。
usecols : 列表或 callable，用于指定要读取的列。可以是列名的列表，也可以是
列号的列表。
na_values : 字符串、列表或字典，用于指定哪些其他值应该被视为 NA / NaN 。
parse_dates : 列表或字典，用于指定将哪些列解析为日期。
skiprows : 整数或列表，用于指定要跳过的行号或条件。
nrows : 整数，用于指定要读取的行数
'''

import pandas as pd

data = pd.read_csv('./人员信息.csv')
print(data)

'''
read_excel
pandas.read_excel 是pandas库中用于读取Excel文件（ .xls 或 .xlsx ）的函
数。它可以将Excel文件中的数据读取为DataFrame对象，便于进行数据分析和处理。


pandas.read_excel(io, sheet_name=0, header=0, index_col=None, 
usecols=None, squeeze=False, dtype=None, skiprows=None, 
nrows=None, na_values=None, keep_default_na=True, 
parse_dates=False, date_parser=None, skipfooter=0, 
convert_float=True, **kwds)

io : 文件路径或文件对象。这是唯一必需的参数，用于指定要读取的Excel文件。
sheet_name=0 : 要读取的表名或表的索引号。默认为0，表示读取第一个工作
表。可以指定工作表名或索引号，如果指定多个，将返回一个字典，键为工作表
名，值为对应的DataFrame

header=0 : 用作列名的行号，默认为0，即第一行作为列名。如果没有标题行，
可以设置为None。
index_col=None : 用作行索引的列号或列名。默认为None，表示不使用任何列
作为索引。可以是一个整数、字符串或列名的列表。
usecols=None : 要读取的列。默认为None，表示读取所有列。可以是一个整数
列表、字符串列表或Excel列的位置（如 [0, 1, 2] ）或字母标记（如 ['A', 
'B', 'C'] ）。
squeeze=False : 如果读取的数据只有一列，当设置为True时，返回一个Series
而不是DataFrame。
dtype=None : 指定某列的数据类型。默认为None，表示自动推断。可以是一个
字典，键为列名，值为NumPy数据类型。
skiprows=None : 要跳过的行号或行号列表。默认为None，表示不跳过任何行。
可以是整数或整数列表。
nrows=None : 读取的行数，从文件头开始。默认为None，表示读取所有行。
na_values=None : 将指定的值替换为NaN。默认为None，表示不替换。可以是
一个值或值的列表。
keep_default_na=True : 如果为True（默认），则除了通过 na_values 指定的
值外，还将默认的NaN值视为NaN。
parse_dates=False : 是否尝试将列解析为日期。默认为False。可以是一个布尔
值、列名列表或列号的列表。
date_parser=None : 用于解析日期的函数。默认为None，表示使用pandas默认
的日期解析器。
skipfooter=0 : 要跳过的文件底部的行数。默认为0，表示不跳过任何底部的
行。
convert_float=True : 是否将所有浮点数转换为64位浮点数。默认为True，以
避免数据类型推断问题。
**kwds : 允许用户传递其他关键字参数，这些参数可能会被引擎特定的读取器所
识别
'''