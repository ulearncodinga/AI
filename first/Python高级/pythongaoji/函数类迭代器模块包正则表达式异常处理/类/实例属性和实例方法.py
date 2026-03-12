"""
实例属性:属于对象的属性,和self参数绑定且多在构造函数中定义

"""
# class Person:
#     a = 1
#     def __init__(self,name,age,gender):
#         self.name = name
#         self.age = age
#         self.gender =gender
# person =Person('zhangsan','18','nan')
#
# # __dict__:用来查看对象的属性,但是只能查看和self绑定了的属性
# print(person.__dict__)
# #查看类属性,还能看类的行为
# print(Person.__dict__)
#实例方法
#类中的函数,第一个参数时self的就是实例方法,就是对象调用方法
class Person:
    def __init__(self,name):
        self.name =name
    def change_name(self,new_name):
        self.name =new_name
    def say(self):
        print(f"我叫{self.name}")

person =Person('zhangsan')
person.say()
person.change_name('lisi')
person.say()



# class Circle:
#     def __init__(self,radius):
#         self.pi = 3.1415926
#         self.radius = radius
#     def area(self):
#         print(self.pi * self.radius * self.radius)
#     def length(self):
#         print(2*self.pi*self.radius)
# c1 = Circle(10)
# c1.area()
# c1.length()