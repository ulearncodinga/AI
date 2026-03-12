"""
自定义迭代器在Python中也可以自定义自己的迭代器，只需要在构造类时实现__iter()__和__next()__两个方法即可，__iter()__和__next()__两个方法的定义如下所示：
"""
class MyRangeIterator:  #定义一个名为MyRangeIterator的类
   def __init__(self, end):  #构造函数，初始化迭代器,end为结束值
       self.end = end  #设置迭代器的结束值
       self.num = 0  #设置当前迭代的数值，初始为0
   def __iter__(self):  #实现__iter__方法，返回迭代器本身
       return self  #因为我们的迭代器就是实例本身，所以返回self
   def __next__(self):  #实现__next__方法，返回下一个值
       if self.num<self.end:  #如果当前数值小于结束值
           value = self.num  #获取当前数值作为要返回的值
           self.num += 1  #将当前数值加1,为下一次迭代做准备
           return value  #返回当前数值
       else:  #如果当前数值已经达到或超过结束值
           raise StopIteration
#抛出StopIteration异常，结束迭代
# #使用MyRangeIterator类创建一个迭代器实例，迭代0到4的整数
for i in MyRangeIterator(5):  # for循环会自动调用迭代器的__next__方法
   print(i)  #打印迭代器返回的每个值

"""
斐波那契数列
定义:
前两个数是0,1,1,2,3,5,8,...
使用迭代器去生成指定个数的斐波那契数列

"""
class FibonacciIterator:#定义一个名为FibonacciIterator的类
    def __init__(self,n):#构造函数,初始化迭代器
        self.n = n#设置迭代器的结束值
        self.a,self.b = 0,1#初始化斐波那契数列的前两个数
        self.count = 0#初始化计数器,用于跟踪迭代次数

    def __iter__(self):#实现__iter__方法,返回迭代器本身
        return self#因为我们的迭代器本身就是实例对象,所以返回self

    def __next__(self):#实现__next__方法,返回下一个值
        if self.count < self.n:#如果当前计数器小于结束值
            value = self.a #获取当前斐波那契数列作为要返回的值
            self.a,self.b = self.b,self.a+self.b#更新斐波那契数列的下一个数
            self.count += 1#增加计数器
            return value#返回当前斐波那契数列
        else:#如果当前计数器已经到达或超过结束值
            raise StopIteration#手动抛出StopIteration异常,结束迭代
#使用FibonacciIterator类创建一个迭代器的实例对象,迭代生成前10个斐波那契数列
fib_iterator = FibonacciIterator(10)#创建一个生成前10个斐波那契数的迭代器
for number in fib_iterator:#for循环会自动调用迭代器的__next__方法
    print(number)