"""# 类的定义 :在python中,使用class关键字来定义一个类
# 注意:使用大驼峰命名法
# class 类名:
    # pass
# 类的作用就是生产对象,解决问题是依靠对象解决
# 类的调用,也叫类的实例化 ,实例化的内容就是对象,该过程就是创建对象
# 创建对象的格式
# 一个类可以创建很多个对象,类创建的对象都是独立的个体
# 被创建的对象有类中的所有东西
# 对象名() = 类名()
"""
#类的定义
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
print(person)#地址
print(Person)#类
print(id(Person))#类的地址
#通过对象去调用类的属性和方法时
#使用对象名字.属性名
print(person.name)
print(person.age)
print(person.gender)
person.eat()#会调用eat函数


#把person这个对象中的属性或行为进行修改
person.name = 'lisi'
print(person.name)
person.age = 20
print(person.age)
#添加一个属性
person.height = 180
print(person.height)
#删除属性
# del person.gender
# print(person.gender)#这时会报错,因为被删除了访问不到