"""
类的封装:
就是将 数据和函数 封装到一个类中，并且通过设置 访问权限 可以将类的属性和方法隐藏在类的内部，
可以防止类的属性和方法被外部直接访问和修改，提高了代码的安全性和可维护性。
"""
"""
在Python，可以通过给属性名和方法名添加下划线(_)设置权限，
又根据下划线的个数分为不同的类型：

单下划线前缀：(约定俗成的)
以单下划线为开头命名的属性或方法，表示该属性或方法是 内部使用 的，
但这是人为规定的，事实上程序是可以访问的。

双下划线前缀：(真正的)
真正的设置私有权限的方法，通过在属性名或行为名之前添加
双下划线就可以将该属性或行为定义为 私有的，且不可被对象访问 。

双下划线前后缀：(特殊的)
表示Python中的 特殊的属性或方法 ，是Python内置好的内容，有特殊的意义和用途，不推荐自己定义。
"""
#私有属性的设置:在属性名和方法名前添加双下划线前缀
class Person:
    def __init__(self,name,password):
        self.name = name
        self.phone_id = '155xxxxxx78'
        self.__password = password#在password前添加__双下划线可以让属性隐藏,访问不到
    def __say(self):
        print(f'我的银行卡密码是{self.__password}')
    def change_password(self):
        self.__say()#这样用没有双下划线的函数调用有双下划线的函数则可以看到
p1 = Person('zhangsan','123456')
print(p1.name)
print(p1.phone_id)
p1.change_password()
