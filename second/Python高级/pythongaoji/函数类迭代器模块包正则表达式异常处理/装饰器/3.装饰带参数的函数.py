"""

"""
#装饰只有一个参数的函数
import time
def timer(f):
    def decorator(x):
        start =time.time()
        ret = f(x)
        print(time.time() - start)
        return ret
    return decorator
@timer
def my_func(x):
    time.sleep(x)

my_func(1)
#装饰位置参数数量的函数
import time
def timer(f):
    def decorator(*args,**kwargs):
        start = time.time()
        ret = f(*args,**kwargs)
        print(time.time() - start)
        return ret
    return decorator#返回内函数,功能的添加也要在内函数中完成
@timer
def my_func(x):
    time.sleep(x)

@timer
def add(x,y):
    time.sleep(1)
    time.sleep(x+y)
my_func(1)
add(2,3)