"""
构造函数与析构函数：
构造函数__init__()：
在创建对象时自动调用的一种函数，用来进行初始化属性，不是必须要定义的，可以在需要的时候再定义。
析构函数__del__()：(一般不会用,python有自己垃圾回收机制)
在对象的引用清零时会自动调用的一种函数，一般用来进行释放资源的操作。不推荐使用，因为Python有自己的垃圾回收机制。
"""
"""
类属性：
在类定义时所定义的属性，用来说明类本身.对象创建时不能修改
实例属性：
属于对象本身的属性，存在于__init__函数中，
在对象创建时会赋值给该对象。
实例方法：(为对象服务,必定有self参数)
第一个参数传入self参数，没有实例化前，无法调用，
实例化后，也多使用对象来调用而不是用类来调用。
静态方法：(自己接收自己的参数)
与类里的数据不共享，有自己的参数，类和对象都可以调用。
类方法：(跟实例方法一样,传入cls就是类方法,传入self就是实例方法)
第一个参数传入cls参数，表示类本身，类和对象都可以调用，用于访问类本身的属性。
"""
class Person:
    #把想要构建的属性放在这个构造init函数的参数里
    def __init__(self,name,age):#在创建对象时是自动调用的,相当于person1,person2创建时被调用了两次,记得把传参进去数据留下来:name,age
        self.name = name#这样就实现了把传递进来的参数绑定在对象中#等价于person.name = '  '
        self.age = age
    def say(self):
        print(f"我叫{self.name},今年{self.age}岁了")

person1 = Person('zhangsan',18)
person2 = Person('lisi',50)

print(person1.name)
print(person2.age)

person1.say()
person2.say()


class Person:
    def __init__(self,name,age):
        self.name = name
        self.age = age
        print("我是构造函数,我被调用了")
    def __del__(self):
        print("我是析构函数,我被调用了")
person = Person('zhangsan',18)

