"""
类:(容器):变量,方法


        对象和类的关系:
            1.对象有类的全部属性和方法
            2.对象的属性和行为可以进行增 删 改
            3.对象不能单个出现,必须依托于类的实例化,一个类可以创建多个对象
            4.对象之间的类属性共享,方法
"""
#类的定义
class car:
    #类属性
    ncolor = None
    nspeed = None
    #类方法
    def setSpeed(self):
        pass
    def setColor(self):
        pass
#对象的创建  类的实例化
#变量名 = 类名()
lbjl = car()

lbjl.ncolor = 'red'
lbjl.nspeed = 100

print(lbjl.ncolor)
print(lbjl.nspeed)
"""增加对象属性"""
lbjl.price = 100000000
print(lbjl.price)

"""删除对象属性 del"""
#
# del lbjl.price
# print(lbjl.price)

"""增加对象的函数"""
from types import MethodType
def mySetPrice(self,price):
    print(price)
    #第一个参数为要加的函数,第二个参数为要加入的函数所在对象
lbjl.mySetPrice = MethodType(mySetPrice,lbjl)

lbjl.mySetPrice(100)


lbjl = car()

car.ncolor = 'red'
car.nspeed = 100

bsj =car()
print("类属性",car.ncolor)
"""
类的方法的调用:类.方法
"""
car.setSpeed(bsj)

class Car():
    ncolor = 'white'
    ntype = 's'
    def Speed(self):
        print("调用了Speed")
    def CColor(self):
        print("调用了CColor")

audi = Car()
print(audi.ncolor)
print(audi.ntype)
