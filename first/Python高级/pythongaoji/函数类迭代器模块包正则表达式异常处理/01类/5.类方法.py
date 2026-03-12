"""
类方法: 第一个参数为cls 同时且装饰器为@classmethod  那就是类方法
调用:类名.方法
应用场景:修改类属性
"""
class Person:
    #类属性
    name = '张三'
    #创建类方法
    @classmethod
    def changeName(cls,name):
        cls.name = name


print(Person.name)
Person.changeName("李四")
print(Person.name)