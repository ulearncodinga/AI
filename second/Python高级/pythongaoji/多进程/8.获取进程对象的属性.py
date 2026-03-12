from multiprocessing import Process
import time
import os#通过os可以查看当前进程的pid和父进程的pid
def say():
    print('child PID:',os.getpid())#获取子进程pid
#或者可以用print('child PID',current_process().pid)
    print('parent PPID:',os.getppid())#获取父进程pid
    time.sleep(3)
if __name__ == '__main__':
# 进程的创建只能在这个主模块下面
    p1 = Process(target=say)#p1是进程对象(必须要有这一行)
    p1.start()#在子进程执行完成之前要让父进程等待(必须要有这一行)
    print('main PID:',os.getpid())#通过这个函数来获取当前进程的pid
    p2 = Process(target=say)
    p2.start()
    # print(p1.name)#获取名字
    # print(p1.daemon)#看进程是否为守护进程
    # print(p1.pid)#查看pid
    # print(p1.exitcode)#查看进程是否结束
    p1.join()#会判断子进程有没有完成,完成之前会阻塞等待(必须要有这一行)
    # print(p1.exitcode)
    p2.join()