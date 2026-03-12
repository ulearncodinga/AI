"""
__init__在创建对象时自动调用,目的为了初始化属性
析构甘薯:__del__ 在对象引用计数为0时自动调用,被del时自动调用,目的释放资源
"""
#类.变量名  类属性
#类.方法名   类方法
#对象.方法  实例方法
#对象,变量名  实例属性
class nTime:
    def __init__(self,hour,minute,second):
        #对象.变量 == 实例属性
        self.hour = hour
        self.minute = minute
        self.second = second
    def __del__(self):#析构函数
        pass
#创建对象 变量名(对象名) = 类名()
mytime = nTime(1,2,3)
print(mytime.hour)
print(mytime.minute)
print(mytime.second)

#对象手动调用构造函数  对象名.函数()
mytime.__init__(17,42,34)
print(mytime.hour)

"""
               实例属性             类属性
定义方式        self.属性名         类中定义的变量
内存共享判断     不共享              共享  
访问方式        对象.变量           类.变量
使用场景        每个属性独有的         对于多少实例共有的属性
"""