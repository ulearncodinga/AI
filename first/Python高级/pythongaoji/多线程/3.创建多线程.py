'''
在Python中，多线程是通过 threading模块实现的，该模块允许程序员创建、启
动、同步多个线程，并提供了一系列的API来支持线程间同步和通信。
在Python中，使用threading模块中的Thread类来创建线程，Thread类创建对象时
的参数为：
threading.Thread(group=None, target=None, name=None, args=(),
kwargs=None, *, daemon=None)
'''

#各个参数说明
'''
group: 应该始终为 None，保留供未来扩展使用。
target: 是一个可调用的对象（函数），该线程启动时，这个对象将被调用。如
果不提供，则不会运行任何东西。
name: 线程名称。默认情况下，将分配一个唯一的名称。
args: 传递给 target函数的位置参数，默认是一个元组。
kwargs: 传递给 target函数的关键字参数，默认是一个字典。
daemon: 指定线程是否为守护线程。如果设置为 True，则该线程会在主线程结
束时自动退出。如果是None，则继承自创建它的线程。
'''
'''
以下是一些Thread类提供的主要方法和属性：
start()：启动线程活动。使用start去启动线程，会调用run()方法，它会创建一
个新的线程来执行run()方法中的代码。
run()：表示线程活动的方法。可以在子类中重写此方法，重写之后执行重写的代
码。通常不需要直接调用run方法，应该调用start方法去启动线程，如果直接调
用run方法(没有通过start方法去启动线程)那么run方法中的代码将在当前线程中
同步执行，而不是在新的线程中执行。
join(timeout=None)：等待线程终止。 timeout参数是可选的，表示等待的最
长时间（以秒为单位）。如果没有指定timeout，则该方法将无限期等待。
is_alive()：返回线程是否还活着。
getName()：返回线程名。
setName(name)：设置线程名。
isDaemon()：返回线程的守护状态。
setDaemon(daemonic)：设置线程的守护状态。必须在start开始前设置。
name：线程名称。
ident：线程的标识符。如果线程尚未启动，则为 None。
daemon：线程的守护状态
'''


# import threading
#
# def func():
#     print("hello")
#
# t1 = threading.Thread(target=func,)
# t1.start()
# t1.join()
# 简化写法
# threading.Thread(target=func,).start()
#
#
#
#
#
# import  threading
# import time
# def func():
#     time.sleep(1)
#     print('helllll')
#
# if __name__ == '__main__':
#     t1 = threading.Thread(target=func,)
#     t1.start()
#
#     print(t1.is_alive())#返回的线程是否还活着
#     t1.join()
#     print(t1.is_alive())


"""
创建线程:
    threading
Tread()参数
    target:执行任务的目标名
    name:进程名 一般不用设置
    group:进程组 目前只能使用None
    args:以元组的方式给执行任务传参
    kwargs:以字典的方式给执行任务传参
"""
#创建不传参线程
#例子1
# from threading import Thread
# import time
# def coding():
#     for i in range(3):
#         print("coding")
#         time.sleep(0.2)
#
# def music():
#     for i in range(3):
#         print("music")
#         time.sleep(0.2)
# if __name__ == '__main__':
#     coding_thread = Thread(target=coding,)
#     music_thread = Thread(target=music,)
#     coding_thread.start()
#     music_thread.start()



#例子2


# import time
# from threading import Thread
#
# def func1():
#     print("Func1 is runing")
#     time.sleep(2)
#     print("Func1 has finished")
# def func2():
#     print("Func2 is running")
#     time.sleep(2)
#     print("Func2 has finished")
# if __name__ == '__main__':
#     start = time.time()
#     process = Thread(target=func2,)
#     process.start()
#     func1()
#     process.join()
#     total_time = time.time()-start
#     print(f"Total time is {total_time}time")



#创建传参进程
# from threading import Thread
# import time
#
# a = 0
# def coding(num,name):
#     for i in range(num):
#         global a
#         print("coding")
#         print(name)
#         a+=1
#         print("coding:",a)
#         time.sleep(0.2)
# def music(count):
#     for i in range(count):
#         print("music")
#         print("music:",a)
#         time.sleep(0.2)
#
# if __name__ == '__main__':
#     coding_thread = Thread(target=coding,kwargs={"num":3,"name":"张三"})
#     music_thread= Thread(target = music,args=(3,))
#     coding_thread.start()
#     music_thread.start()


#通过自定义类的方式创建多进线程
# from threading import Thread
# import time
# class MyThread(Thread):
#     def __init__(self,name):
#         super().__init__()
#         self.task_name = name
#     def run(self):
#         print(f"{self.task_name}任务开始")
#         time.sleep(2)
#         print(f"{self.task_name}任务结束")
#
# if __name__ == '__main__':
#     p = MyThread("测试")
#     p.start()
#     print("主进程")



'''
2.3 Timer
Timer，也被称为定时器，它允许你在一定时间后执行一个函数或者可调用的对象。
Timer类是 Thread类的一个子类，因此它具有线程的所有特性，并且可以用来在后
台执行定时任务。
以下是Timer类的基本用法和特性：
class threading.Timer(interval, function, args=None, kwargs=None)



参数说明:
interval: 一个浮点数或整数，表示在执行 function之前需要等待的时间（以秒
为单位）。
function: 一个可调用的对象，当定时器到期时将被执行。
args: 传递给 function的位置参数元组。
kwargs: 传递给 function的关键字参数字典

 实例方法
start(): 启动定时器。 Timer将在指定的时间间隔后开始执行目标函数。
cancel(): 取消定时器。如果定时器尚未启动，则此方法无效。如果定时器正在运
行，调用 cancel()将停止定时器，并且目标函数不会被调用。
'''
#获取线程的信息
#
# import os
# from threading import Thread,current_thread
# import time
#
# def coding():
#     print(f"进程号: {os.getpid()}")
#     print(f"线程信息:{current_thread()}")
#     for i in range(3):
#         print("coding")
#         time.sleep(0.2)
# def music():
#     print(f"music:{os.getpid()}")
#     print(f"线程信息:{current_thread()}")
#     for i in range(3):
#         print("music")
#         time.sleep(0.2)
# if __name__ == '__main__':
#     print(f"主进程为:{os.getpid()}")
#     cod_thread = Thread(target = coding)
#     music_thread = Thread(target=coding)
#     cod_thread.start()
#     music_thread.start()

"""
主线程和子线程的结束顺序
1.两个进程里面的两个线程 主进程和子进程结束顺序:默认情况下
主进程会等待所有子进程结束完之后才会结束
2.一个进程里面的两个线程 主线程和子线程的结束顺序:默认情况下
主线程会等待所有子线程结束完之后才会结束
"""
# from threading import Thread
# import time
# def work():
#     for i in range(5):
#         print("工作1")
#         time.sleep(0.2)
# if __name__ == '__main__':
#     work_thread = Thread(target=work)
#     work_thread.start()
#     time.sleep(1)
#     print("主进程执行完毕")



#线程间的执行顺序:无序
# from threading import Thread,current_thread
# import time
#
#
# def get_info(a):
#     time.sleep(0.5)
#     print(a)
#     print("get_info",a)
#     print("get_info",current_thread)
# if __name__ == '__main__':
#     for i in range(5):
#         process = Thread(target=get_info,args=(i,))
#         process.start()


import queue
# 创建一个 PriorityQueue
priority_queue = queue.PriorityQueue(5)
# 向 PriorityQueue 中放入元素
priority_queue.put((2, 'Task1'))
priority_queue.put((0, 'Task2'))
priority_queue.put((1, 'Task3'))
# 从 PriorityQueue 中取出元素
print(priority_queue.get())
print(priority_queue.get())
print(priority_queue.get())













