"""
静态方法:有自己的参数,与类里的数据不共享,类和对象都可以被调用
@staricmethod
"""
class Calculator:
    @staticmethod
    def add(a,b):
        print(a + b)
Calculator.add(1,2)
cal = Calculator()
cal.add(3,4)

#类方法:属于类本身的方法,用来访问和修改类的属性,不能访问实例属性
class Person:
    name = 'zhangsan'

    @classmethod
    def change_name(cls,name):
        cls.name =name
        print(cls)
print(Person)
person =Person()
person.change_name('lisi')
print(person.name)

import logging
logging.basicConfig(filename='./app.log',level=logging.DEBUG,filemode='a',format = '%(name)s - %(levelname)s - %(message)s')
print("这是一个警告信息")