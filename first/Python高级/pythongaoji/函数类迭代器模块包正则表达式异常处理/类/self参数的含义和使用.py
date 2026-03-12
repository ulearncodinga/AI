"""
self：(相当于是对象的身份证,存的就是地址)
是一个参数，表示对象自身，里面存放着对象自身的地址。如果希望类中的方法可以被对象调用，那么第一个参数必须是self。
其作用就是将实例对象与类的方法进行绑定，这样才能让每个对象都能调用“属于”自己的方法。
"""
class Dog:
    #两个用法
    def speak(self):
         print(f"{self}在叫")

dog1 = Dog()
dog1.speak()
dog2 = Dog()
dog2.speak()

#self参数
class Person:
    def say(self):
        print(self)
person1 = Person()
person2 = Person()
person1.say()
print(person1)
person2.say()
print(person2)
