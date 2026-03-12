#嵌套函数
#嵌套函数的作用域:内部函数的局部变量是可以覆盖外部函数的局部变量的
#在内函数中:内函数的局部变量>外函数的局部变量>全局变量
#内部函数除了在外部函数中调用之外,是无法访问到的
def outer_func():
    def inner_func():
        print('这是内部函数')
    print('这是外部函数')
    return
outer_func()

def outer_func():
    x = 1
    def inner_func():
        #使用nonlocal关键字,可以在内函数中修改外函数的变量值
        #global和nonlocal的区别:
        #global 修改的是局部变量,nonlocal修改的是上一层嵌套函数的变量
        #global和nonlocal不能对同一个变量生效
        nonlocal x #将这个x=2把外面的也改了
        x =2
        print('在内部函数中x的值:',x)
    inner_func()
    print('在外函数中的x的值:',x)
    return
outer_func()

#以内函数作为返回值进行返回
def outer_func():
    def inner_func():
        print("这是一个内函数:")
    inner_func()
    print("这是一个外函数")
    return inner_func
ret = outer_func()
ret()

#嵌套函数的特性:当内部函数访问外部函数的某个变量时
#改变量不会随着外部函数的调用完毕而销毁,而是被内部函数所保留
def outer_func():
    x =1
    def inner_func():
        print(x)
    return inner_func

ret = outer_func()
ret()


#pow函数幂运算
#方法一
def pow(x,y):
    return x**y
pow(3,4)

#方法二
def outer_func(exp):
    def inner_func(base):
        return base ** exp
    return inner_func
ret = outer_func(2)#exp   平方
result = ret(3)#base
print(result)
res = ret(4)
print(res)

ret1 = outer_func(3)#三次方
res1 = ret(10)
print(res1)