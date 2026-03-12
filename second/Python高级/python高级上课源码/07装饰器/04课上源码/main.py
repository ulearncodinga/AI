

# def decorator(f):
#     def f1():
#         print('b')
#         f()
#         print('c')
#     return f1
#


# def func():
#     print('a')
#
#
# func = decorator(func)
# func()

#
# @decorator
# def func():
#     print('a')
#
#
# func()




# import time
#
# def timer(f):
#
#     def decorator(x):
#         start = time.time()
#         ret = f(x)
#         print(time.time() - start)
#         return ret
#     return decorator
#
# @timer
# def my_func(x):
#     time.sleep(x)
#
#
# my_func(1)


#
# import time
#
# def timer(f):
#
#     def decorator(*args, **kwargs):
#         start = time.time()
#         ret = f(*args, **kwargs)
#         print(time.time() - start)
#         return ret
#     return decorator
#
#
# @timer
# def my_func(x):
#     time.sleep(x)
#
# @timer
# def add(x, y):
#     time.sleep(1)
#     print(x + y)
#
# my_func(1)
# add(2, 3)







# def func1(func, prefix):
#     def decorator(*args, **kwargs):
#         result = func(*args, **kwargs)
#         return f"{result} {prefix}"
#     return decorator
#
# # 使用装饰器并传递参数
# @func1(prefix="How are you?")
# def greet(name):
#     return f"Hello, {name}."
#
# greet = func1(greet, 'How are you?')
#
# # 调用被装饰的函数
# print(greet("Alice"))




#
#
# def decorator_one(func):
#     def wrapper_one(*args, **kwargs):
#         print("Decorator one - before call")
#         result = func(*args, **kwargs)
#         print("Decorator one - after call")
#         return result
#     return wrapper_one
#
# def decorator_two(func):
#     def wrapper_two(*args, **kwargs):
#         print("Decorator two - before call")
#         result = func(*args, **kwargs)
#         print("Decorator two - after call")
#         return result
#     return wrapper_two
#
# @decorator_one
# @decorator_two
# def say_hello(name):
#     print(f"Hello, {name}")
#
# # say_hello = decorator_one(decorator_two(say_hello))
# say_hello("Alice")





# class MyDecorator:
#     def __init__(self, func):
#         self.func = func
#
#
#     def __call__(self, *args, **kwargs):
#         print("Something is happening before the function is called.")
#         result = self.func(*args, **kwargs)
#         print("Something is happening after the function is called.")
#         return result
#
# @MyDecorator
# def say_hello(name):
#     print(f"Hello, {name}")
#
# say_hello("Alice")






# import logging
#
# def log_decorator(func):
#     def wrapper(*args, **kwargs):
#         logging.basicConfig(filename='./app.log', level=logging.INFO, filemode='a', format='%(name)s - %(levelname)s - %(asctime)s- %(message)s')
#         logging.warning(f"Calling function: {func.__name__} with args: {args} and kwargs: {kwargs}")
#         result = func(*args, **kwargs)
#         logging.warning(f"Function {func.__name__} returned: {result}")
#         return result
#     return wrapper
#
#
# @log_decorator
# def test():
#     print('123')
#
# test()






class User:
    def __init__(self, username, role):
        self.username = username
        self.role = role

# 装饰器：检查用户是否具有管理员权限
def admin_only(func):
    def wrapper(user, *args, **kwargs):
        if user.role != 'admin':
            raise PermissionError(f"User {user.username} is not authorized to perform this action.")
        return func(user, *args, **kwargs)
    return wrapper

# 使用装饰器的函数
@admin_only
def delete_user(user, user_to_delete):
    """只有管理员可以删除用户"""
    print(f"User {user_to_delete.username} has been deleted by {user.username}.")

# 测试代码
admin_user = User('Alice', 'admin')
normal_user = User('Bob', 'user')

# 尝试以管理员身份删除用户
delete_user(admin_user, normal_user)  # 应该成功

# 尝试以普通用户身份删除用户
try:
    delete_user(normal_user, admin_user)  # 应该失败
except PermissionError as e:
    print(e)








