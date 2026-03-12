#1.创建不传参的进程
#查看各进程的pid以及ppid
import os
from multiprocessing import Process

def func1():
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"执行func2的pid是:{my_pid}")
    print(f"执行父进程的ppid是:{ppid}")

def func2():
    my_pid = os.getpid()
    ppid = os.getppid()
    print(f"执行func1的pid是:{my_pid}")
    print(f"执行func1的ppid是:{ppid}")
if __name__ == '__main__':
    p1 = Process(target=func1)
    p1.start()
    p2 = Process(target=func2)
    p2.start()

    current_pid = os.getpid()
    print(f"主进程的PID为:{current_pid}")

    p1.join()
    p2.join()




# 运行时间统计
import time
from multiprocessing import Process

def func1():
    print("Function 1 is running.")
    time.sleep(2)
    print("Function 1 has finished.")

def func2():
    print("Function 2 is running")
    time.sleep(2)
    print("Function 2 is finished")

if __name__ == '__main__':
    start_time = time.time()
    process = Process(target=func2)#创建一个子进程来运行func2
    process.start()

    #获取创建进程的PID
    pid = process.pid
    #主进程运行func1
    func1()

    #等待子进程结束
    process.join()
    #计算并且打印整个程序的运行时间
    total_time = time.time() - start_time
    print(f"Total execution time:{total_time:.2f} seconds")

#创建传参进程
#使用args传参
from multiprocessing import Process
import time

def say_hello(name):
    time.sleep(2)
    print(f"Hello {name},Nice to meet you!")

if __name__ == '__main__':
    start = time.time()
    process1 = Process(target=say_hello,args=('Alice',))
    process1.start()
    say_hello('Bob')
    process1.join()
    exe_time = time.time() - start
    print(exe_time)


#全局变量不共享

from multiprocessing import Process
import time
moneny = 100
def expend(moneny):
    moneny -= 20
    print('child PID',moneny)
if __name__ == '__main__':
    process1 = Process(target=expend,args=(moneny,))
    process1.start()
    process1.join()
    print('main PID:',moneny)


#kwargs传参
from multiprocessing import Process
import time
def say_hello(name):
    time.sleep(2)
    print(f"Hello{name},Nice to meet you !")
if __name__ == '__main__':
    start = time.time()
    process1 = Process(target=say_hello,kwargs={'name':'Alice'})
    process1.start()
    say_hello('Bob')
    process1.join()
    exe_time = time.time() - start
    print(exe_time)