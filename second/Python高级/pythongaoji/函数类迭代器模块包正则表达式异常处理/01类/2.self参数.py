"""
    self:表示对象本身,里面存放着对象的地址
    作用:将对象与类方法进行绑定,这样每个对象都能调用自己的方法
    属于自己的方法(实例方法)
    """

class Dog:
    def Speak(self):
        print(f"{self}在说话")
        self.func2()
    def func2(self):
        print("func2")
dog1 =Dog()
dog1.Speak()#对象.方法() 自动将对象本身传入self
dog1.abc =1
# dog2 = Dog()
# dog2.Speak()