"""
类中属性和方法的类别
1.类属性:在类定义时所定义的属性,不和self绑定
"""
#如果类的所有实例都需要使用同一个数据时,就可以把这个变量设置成类属性
#计算圆的面积
class Circle:
    PI = 3.1415926

    def __init__(self,radius):
        self.radius = radius
    def area(self):
        print(f"圆的面积是:{Circle.PI*self.radius*self.radius}")
c1 = Circle(1)
c2 = Circle(2)
c1.area()
c2.area()

#模拟银行账户
class Bank:
    total_amount = 0
    def __init__(self,name,init_money):
        self.name = name
        Bank.total_amount += init_money#开户时存了多少钱
    def save_money(self,amount):
        Bank.total_amount += amount#存钱
    def expend_money(self,amount):
        Bank.total_amount -= amount#取钱

my_bank1 = Bank('zhangsan',10000)
my_bank1.save_money(20000)#存20000
my_bank1.expend_money(5000)#取5000
print(f"{my_bank1.name}的账户还有{Bank.total_amount}钱")