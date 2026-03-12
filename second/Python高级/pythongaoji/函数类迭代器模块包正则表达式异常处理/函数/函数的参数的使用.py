"""
位置不定长参数 *args
关键字不定长参数 ** kwargs(标志)
它会将输入的关键字参数中的关键字当作键值对的键,将关键字参数中的实参当作键值对的值放在字典中进行存储
"""
def func(**kwargs):
    print(kwargs)
    print(type(kwargs))
func(name = 'zhangsan',age = 18)
#对传入的关键字参数的值进行相乘
def mul(**kwargs):
    result = 1
    #使用kwargs.keys()获取字典中的所有的键
    #通过for循环遍历这些键,从而获取对应的值并令该值参与运算
    for key in kwargs.keys():
        result = kwargs[key] * result
    print(result)

mul(a = 2,b = 3,c = 6,d = 10)

"""
args和kwargs出现在同一个函数中
kwargs中的关键字的参数,不能和其他关键字重名
"""
def test(x,*args,**kwargs):
    print(x)
    print(args)
    print(kwargs)
test(1,2,3,z = 4,y = 5)#这里的z就不能取x要不然会报错TypeError

"""
元组的解包用法,使用*元组名就可以做到一次性传递多个参数
"""
def test3(x,y,m,n):
    print(x,y,m,n)
args = (1,2,3,4)
test3(*args)#*号不能少,解包功能


kwargs = {'x':1,'y':2,'m':3,'n':4}#这里不能用别的只能跟函数形参一样用xymn
#需要注意键值对的键必须要和函数形参的名称相同才能传递参数
test(**kwargs)
