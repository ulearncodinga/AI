def add(x,y):
    return x + y
def sub(x,y):
    return x - y
def mul(x,y):
    return x * y
def div(x,y):
    return x / y



__all__ =['add','sub']#写在这里面的可以被导入,如果不把div和mul写在__all__里这两个就不能被导入


a = 1
b = 2


# if __name__ == '__main__':
#     ret = add(5,6)
#     print(ret)
#怎么才能让这两行代码在调用时不被有运行,要加入if __name__ == '__main__':
#把要主动执行的代码放在这个里面if __name__ == '__main__':

