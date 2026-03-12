"""
对象与类之间的关系：
对象拥有类所定义的全部的属性和行为。
对象的属性和行为可以单独进行增加、删除、修改。
对象不能单独创建，必须依托类的实例化，且一个类可以实例化无数的对象。
对象之间的属性和行为是不可以共享的
python中一切皆对象object
"""
#对方法(函数)的添加和修改
class Person():
    name = 'zhangsan'
    age = 18
    gender = 'nan'

    def eat(self):
        print('我会吃饭')
    def drink(self):
        print('我会喝水')
#创建对象
person = Person()
def drink(self):
    print("我要喝饮料")
    #需要导入一个库,来将定义的函数与对象绑定起来
from types import MethodType
person.drink = MethodType(drink,person)
person.drink()

def say(self):
    print("我会说话")
person.speak = MethodType(say,person)
person.speak()#调用的时候时要输入这个名字而不是函数名
#删除
# del person.speak
# person.speak()