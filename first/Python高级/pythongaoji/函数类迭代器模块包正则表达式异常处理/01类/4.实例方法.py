"""
        实例方法是什么:第一个参数传入self参数,那么该函数就是实例方法
        调用: 对象名.方法

"""

class Person:
    def __init__(self,name,age):
        self.name = name
        self.age =age
    def changeAge(self,age):
        self.age = age
person1 = Person("张三",18)
person1.changeAge(10)
print(person1.name,person1.age)
print(person1.age)