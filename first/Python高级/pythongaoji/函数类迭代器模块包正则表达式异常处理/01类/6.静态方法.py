"""
    静态方法: 第一个参数直接就是函数参数列表且装饰器名为staticmethod
"""
class Email:
    a =1
    def __init__(self,email):
        self.email = email
    @staticmethod
    def is_valid_email(Nemail):
        if '@' in Nemail:
            return True
        else:
            return False

    #检查邮箱是否正确