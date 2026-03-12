#局部变量
def test():
    x = 1
    print(x)
    return
test()
#变量x是在函数内部定义的,是局部变量
#仅仅在函数生效时存在,函数执行完毕后就无法访问到该变量


#全局变量:定义在函数的外边的变量或者在函数内部使用global函数关键字修饰的变量
#在函数的内部是可以访问到全局变量的,但函数的外部不能访问到函数内部的变量
#当函数内部存在变量与函数外部变量名字相同的时候,局部变量会覆盖全局变量
#global非必要不使用,因为global能够在函数的内部修改全局变量
#使用很多的global会导致全局变量危险
y = 1
def test1():
    global y#把y=10变成全局变量,并且覆盖了之前的y=1
    print('函数内部修改值之前的id:',id(y))
    y=10
    print('函数内部修改值之前的id:', id(y))
    print(y)
print('函数外部修改值之前的id:',id(y))
test1()
print('函数外部修改值之前的id:', id(y))
print(y)

"""
关于列表的len()函数的实现
"""
def len_list(my_list):
    #使用count来记录元素的个数
    count = 0
    for i in my_list:
        count += 1
    return count
my_list = [1,2,3,'asasas',[1,3]]
ret = len_list(my_list)
print(ret)

"""
关于列表的sum函数
函数的名字或者变量的名字一定不能跟内部函数名字相同,否则调用不到内部函数了
"""
def sum_list(my_list):
    #定义一个变量存储求和的结果
    result = 0
    for i in my_list:
        result += i
    return result
my_list_num = [x for x in range(101)]
ret = sum_list(my_list_num)
print(ret)