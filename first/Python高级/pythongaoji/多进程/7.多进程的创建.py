"""
在Python中，multiprocessing库提供了创建和管理进程的方法，它允许程序员创
建进程，并提供了一系列的API来支持进程间数据共享、同步和通信。
在Python中，使用multiprocessing中的Process类来创建子进程，Process类创建对
象时的参数为：
multiprocessing.Process(group=None, target=None, name=None, args=
(), kwargs={}, *, daemon=None)
"""
'''
参数说明
以下是各个参数的说明：
group：通常不使用，是为将来可能的扩展预留的。
(重要)target：表示调用对象，即子进程要执行的任务。这个参数通常是一个函数的
名字。
name：进程的名称。默认情况下，进程名称为 Process-N，其中 N是进程的序
号。可以通过这个参数来指定一个自定义的名称。
args：表示调用 target函数时传递的参数元组。
kwargs：表示调用 target函数时传递的参数字典。
daemon：如果设置为 True，则子进程将是一个守护进程。当主进程结束时，所
有守护进程都将被终止。默认值为 None，表示继承当前进程的守护进程设置。
'''
'''
创建的对象的方法
创建的对象也拥有一些方法，分别是：
1. start()：启动进程。这将执行在创建 Process对象时指定的 target函数，调
用start时会自动调用run。
2. run()：此方法用于定义进程启动时执行的操作。默认情况下，它调用传递给
target参数的函数。如果 target没有指定，你可以覆盖此方法来实现自定义的
进程行为。
3. join([timeout])：主进程等待子进程终止或直到达到指定的超时时间。如果
timeout被省略或为 None，则主进程将一直停留在这里。
4. is_alive()：返回一个布尔值，表示进程是否仍然在运行。
5. terminate()：强行终止进程，且不会进行任何清理操作。 如果该进程创建了
子进程，那么这个子进程就变成了僵尸进程，如果p还保存了一个锁，那么这个
锁也不会被释放，会变成死锁。
6. kill()：终止进程。在Unix上，这是通过发送SIGKILL信号实现的；在Windows
上，则是通过调用 TerminateProcess。
7. close()：关闭进程。(必须要在join后)此方法释放Process对象所持有的资源，如果子进程仍在
进行，调用此方法将是错误的。
'''
'''
属性:
1. name：返回或设置进程的名称。
2. daemon：返回或设置进程是否为守护进程，如果设置的话，必须在start()之前
设置。
3. pid：返回进程的PID。
4. exitcode：返回进程的退出代码，如果进程未结束，就返回None，负值-N表示
子进程被信号N终止，正常终止返回0。
5. authkey：返回或设置进程间通信的密钥，用于进程间通信的身份认证，了解即
可。
注意：Windows中的Process必须放到 if __name__ == "__main__"下面。
'''
from multiprocessing import Process
import time
def say():
    print('in p1')
    print("Hello")
    time.sleep(3)

if __name__ == '__main__':
# 进程的创建只能在这个主模块下面
    print('in main')
    p1 = Process(target=say)#p1是进程对象
    print('make p1')
    p1.start()#在子进程执行完成之前要让父进程等待
    p1.join()#会判断子进程有没有完成,完成之前会阻塞等待
    print('OK')#这个会在子进程完成后才会执行这一句

"""
创建进程的底层方法:
根据不同的平台，multiprocessing 支持三种启动进程的方法。这些启动方法有
spawn    (Windows只支持这一种)
父进程会启动一个全新的 python 解释器进程。 子进程将只继承那些运行进程对象的 run()方法所必需的资源。 特别地，来自父进程的非必需文件描述符和句柄将不会被继承。 使用此方法启动进程相比使用 fork 或 forkserver 要慢上许多,
可在 Unix 和 Windows 上使用。 Windows 和 macOs 上的默认设置。
fork
父进程使用 os.fork()来产生 Python 解释器分叉。子进程在开始时实际上与父进程相同。父进程的所有资源都由子进程继承。请注意，安全分又多线程进程是棘手的。
只存在于 Unix。 Unix 中的默认值。
forkserver
程序启动并选择 forkserver启动方法时，将启动服务器进程。 从那时起，每当需要一个新进程时，父进程就会连接到服务器并请求它分叉一个新进程。分叉服务器进程是单线程的，因此使用os.fork()是安全的。没有不必要的资源被继承。
可在Unix平台上使用，支持通过Unix管道传递文件描述符。
"""

"""
Process参数
target:函数名
name:进程名
group:进程组
args,以元组的形式给执行任务传参
kwargs:以字典的形式给执行任务传参
"""