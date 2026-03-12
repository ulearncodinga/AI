import re

# 编译一个正则表达式模式，用于匹配手机号
email_pattern = re.compile(r'apple(?! {2}banana)')

# 使用编译后的模式进行多次匹配
text = '''
apple
apple  banana
apple  orange
'''
emails = email_pattern.findall(text)

print(emails)