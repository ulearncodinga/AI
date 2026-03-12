"""
定义一个汽车类Car
(1)在构造方法中添加颜色color、型号type等属性
(2)类中定义一个move方法，方法输出“这辆车的颜色为XX,型号为XX”
(3)分别创建bmw、audi对象.bmw的颜色为red,型号为x1;audi的颜色为blue,型号为a4
(4)分别调用move方法
"""
class Car:
    def __init__(self,color,type):
        self.color = color
        self.type = type
    def move(self):
        print(f"这辆车的颜色为{self.color},这辆车的型号为{self.type}")

bmw = Car('red','x1')

audi = Car('bule','a4')

bmw.move()
audi.move()
bmw.__init__('blue','x7')
print(f"该车颜色为{bmw.color}型号为{bmw.type}")
"""
1.接收用户输入的长方形的长和宽，输出长方形的周长

2.接收用户输入的圆的半径，输出圆的面积.(π用3.14)
"""
# length = float(input("请输入长方形的长:"))
# width = float(input("请输入长方形的宽:"))
# C = (length + width) * 2
# print(f"长方形的周长为:{C}")
# r = float(input("请输入长方形的半径:"))
# pi = 3.14
# S = pi * r * r
# print(f"长方形的面积是:{S}")
"""
三、编程题：输入自己的身份证号码，按下列格式输出自己的出生日期信息
如输入：110101200612260019
输出：我的出生日期是 2006年 12 月 26 日
"""
card_id = '430681200101010059'
year = card_id[6:10]
month = card_id[10:12]
day = card_id[12:14]
print(f"我得出生日期为:{year}年{month}月{day}日")


"""
定义一个汽车类Car
(1)在构造方法中添加颜色color、型号type等属性
(2)类中定义一个move方法，方法输出“这辆车的颜色为XX,型号为XX”
(3)分别创建bmw、audi对象.bmw的颜色为red,型号为x1;audi的颜色为blue,型号为a4
(4)分别调用move方法
"""
class Car:
    def __init__(self,color,type):
        self.color = color
        self.type = type

    def move(self):
        print(f"这辆车的颜色为{self.color},型号为:{self.type}")

bmw = Car('red','x1')
bmw.move()
audi = Car('bule','a4')
audi.move()















































