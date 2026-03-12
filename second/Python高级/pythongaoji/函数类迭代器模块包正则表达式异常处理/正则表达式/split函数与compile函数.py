"""
该函数的作用就是将某文本根据匹配模式进行分割，并将分割后的结果放入列表中返回。
函数原型：
re.split(pattern，string，maxsplit，flags=0)
pattern：正则表达式的格式。
string：被匹配的文本。maxsplit（可选）：这是可选参数，表示最大分割次数。默认值为0，表示分割所有匹配项。
flags（可选）：标志位，用于控制正则表达式的匹配方式，如：是否区分大小写、设置多行匹配模式等，


"""

import re
text = "apple,banana,orange,watermelon"
fruits = re.split(r',\s',text)
print(fruits)

"""

该函数会预先编译正则表达式要匹配的模式，并会返回一个正则表达式的对象，该对象与re.match返回的对象不同,该对象可以调用上面的函数。
函数原型：re.compile(pattern, flags=0)
pattern：要匹配的正则表达式。
flags（可选）：标志位，用于控制正则表达式的匹配方式，
如：是否区分大小写、设置多行匹配模式等，具体有哪些标志可参考下面的

常见的应用场景：
多次匹配：当你需要在一个较长的文本中多次应用同一个正则表达式时，使用re.compile可以避免每次匹配时都重新编译表达式。
"""
import re
email_pattern = re.compile(r'apple(?= banana)')
#使用编译后的模式进行多次匹配
text ='''
apple
apple banana
apple orange

'''

emails = email_pattern.findall(text)
print(emails)