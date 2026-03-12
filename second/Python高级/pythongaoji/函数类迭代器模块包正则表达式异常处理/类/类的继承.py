"""
在Python中，
一个类可以被继承，也可以去继承别的类，其中被继承的类被称为父类或基类，继承的类叫做子类。
在继承过程中，子类会继承父类的所有的属性和行为，并且一个子类可以有多个父类，一个父类可以有多个子类。
(父类的内容一般不建议被修改,不然通过父类创建的子类都会出错)
格式为：
class Person(要继承的类):
    pass
"""
"""
在Python3.x中，所有的类都有一个共同的父类，叫做object类，
即使你在自定义自己的类时，也会默认继承object类，子类会拥有object类的所有的属性和方法。
"""
"""
__new__()   创建对象
__init__()  对对象进行初始化
__eq__()    定义比较操作符==的行为
__iter__()  使对象支持迭代
__next__()  在迭代中返回下一个值
__class__   对象所属的类
__doc__     对象的文档字符串
_name_      类的名称
"""
"""
单继承：
一个类只有一个父类，就是单继承。当子类中存在与父类相同的属性或方法时，子类的属性或方法会覆盖掉父类的属性或方法。

多继承：
一个类有多个父类，就是多继承。多继承可以提供更多的功能，但是也可能导致继承冲突，
比如子类和多个父类有相同的属性和方法。那么此时在使用该属性或方法时，
其顺序为：
            子类>从左到右第一个父类>第二个父类>第三个>…

对于复杂的继承关系，使用子类的mro()方法获取其继承顺序。
"""
#类的单继承
class Zhangsan:
    def __init__(self):
        self.height = 180
        self.weight = 140
        self.age = 18
    def say(self):
        print("我要上大学")
class Lisi(Zhangsan):#只继承了Zhangsan这一个父类
    pass
lisi = Lisi()
print(lisi.height)
print(lisi.age)
lisi.say()




#当子类中存在与父类相同的属性或者方法时,子类的属性或者方法会优先调用自己的属性和方法
#如果自己也没有该属性和方法,才会调用父类中的属性和方法(可重复性大大提高)
class A:
    a = 1
    b = 2
    c = 3
    def say(self):
        print("我是A")
    def question(self):
        print("你是谁?")
class B(A):
    a = 4
    b = 5
    def say(self):
        print("我是B")

b = B()
a =A()
print(b.a)
print(b.b)
print(b.c)
b.say()
a.say()
print(a.a)
print(a.b)
print(a.c)


#类的多继承:多继承的属性和访问属性,是按照继承顺序,从左到右进行访问的
#比如说:调用某个方法时,会首先在自身的类中去寻找,如果自身的类没有这个方法,就会从继承的父类中去寻找
class A:
    a = 100
    def say(self):
        print("A")
    def run(self):
        print("我会跑步")
class B:
    a = 200
    def say(self):
        print("B")
    def swim(self):
        print("我会游泳")
class C(A,B):
    pass

c = C()
print(c.a)
c.say()
c.swim()


class A:
    def __init__(self,x,y):
        self.x = x
        self.y = y

    def add(self):
        return self.x + self.y
class B(A):
    def __init__(self,x,y,z):
        A.__init__(self,x,y)
        self.z = z
    def add(self):
        return A.add(self) + self.z

a = A(1,2)
print(a.add())
b = B(1,2,3)
print(b.add())











class A:
    def __init__(self):
        print('A')
class B1(A):
    def __init__(self):
        A.__init__(self)
        print('B1')
class B2(A):
    def __init__(self):
        A.__init__(self)
        print('B2')
class C(B1,B2):
    def __init__(self):
        B1.__init__(self)
        B2.__init__(self)
        print('C')
c = C()
print("=================")
#super()函数:根据mro()函数继承顺序去搜索父类中指定的函数,并且自动绑定好self函数
class A:
    def __init__(self):
        print('A')
class B1(A):
    def __init__(self):
        super().__init__()
        print('B1')
class B2(A):
    def __init__(self):
        super().__init__()
        print('B2')
class C(B1,B2):
    def __init__(self):
        super().__init__()
        print('C')
c = C()

print("=================")
class A:
    def __init__(self,x,y):
        self.x = x
        self.y = y
        print('A')
    def add(self):
        print(self.x + self.y)
class B(A):
    def __init__(self,a,b):
        super().__init__(a,b)
        print('B')
b =B(1,2)
b.add()