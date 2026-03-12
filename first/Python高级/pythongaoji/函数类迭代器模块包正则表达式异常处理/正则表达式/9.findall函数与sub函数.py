"""
该函数从文本中寻找所有与模式匹配的子串，并将所有的匹配结果存储到一个列表中进行返回，如果没有匹配成功会返回一个空列表。
函数原型：
re.findall(pattern，string，flags=0)
pattern：正则表达式的格式。
string：被匹配的文本。
flags（可选）：标志位，用于控制正则表达式的匹配方式，如：是否区分大小写、设置多行匹配模式等，具体有哪些标志可参考下面的表格。
常用的应用场景有：
1.提取多个子串：当你需要在字符串中找到所有匹配特定模式的字串时。
2.文本分析：在文本中，提取文本中特定的词汇、短语或模式。
"""
#sub函数
import re
text = "1999.12.31是20世纪的最后一天"
pattern = r"(\d{4})\.(\d{2})\.(\d{2})"
replacement = r"\1年\2月\3日"

new_text = re.sub(pattern,replacement,text)
print(new_text)
#进行替换

import re
#原始字符串
text = "Hello 123 World 456"
#定义一个正则表达式,匹配数字
pattern = r'\d+'

#定义一个替换函数
def replace(match):
   #将匹配到的数字乘以2
   number = int(match.group())
   return str(number*2)#使用re.sub和替换函数
new_text = re.sub(pattern, replace, text)
print(new_text)  #输出"Hello 246 World 91

#脱敏
import re
def desensitize_info(text):
   #身份证号的正则表达式
   id_card_pattern = r'(\d{6})(\d{4})(\d{4})(\d{3})(\d|X)'
   #手机号的正则表达式
   phone_pattern = r'(\d{3})(\d{4})(\d{4})'
   #脱敏身份证号
   text = re.sub(id_card_pattern, r'\1********\4\5',text)
   #脱敏手机号
   text = re.sub(phone_pattern, r'\1****\3',text)
   return text
#示例文本
text = "我的身份证号是12345619870605123X，手机号是13812345678"
#脱敏处理desensitized_text = desensitize_info(text)
desensitized_text = desensitize_info(text)
print(desensitized_text)

