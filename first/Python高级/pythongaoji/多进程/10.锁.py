'''
 锁
锁（Lock）是一种用于控制多个进程访问共享资源的机制，锁的主要目的是防止多个
进程同时访问共享资源时可能产生的竞态条件（Race Condition），确保数据的一致
性和完整性。在Python中，最常用的锁为互斥锁（Lock）和递归锁（RLock）。
互斥锁
这是最常见的一种锁，它确保同一时间只有一个进程可以访问共享资源。当一个进程
正在使用资源时，它会锁定该资源，其他进程必须等待锁被释放后才能访问。在
multiprocessing模块中， Lock对象可以用来确保临界区代码的互斥执行。
方法：
acquire(blocking=True, timeout=-1)：尝试获取锁。如果 blocking为
True并且 timeout是默认值 -1，该方法会阻塞直到锁被获取。如果 blocking
为False，则立即返回而不阻塞。

release()：释放锁。
(获取锁后要释放,要不然都访问不了这个资源,是让他们有秩序的去使用,而不是不使用)
'''
# import time
# from threading import Thread,Lock
#
# #100个线程去抢100张票
# ticket_num = 100
# def task():
#     global ticket_num
#     temp = ticket_num
#     time.sleep(0.05)
#     ticket_num = temp - 1
#
# if __name__ == '__main__':
#     tasklist = []
#     for i in range(100):
#         sub_thread = Thread(target=task)
#         sub_thread.start()
#         tasklist.append(sub_thread)
#
#     for task in tasklist:
#         task.join()
#     print("主线程: ",ticket_num)

# from threading import Thread,Lock
# import time
# ticket_num = 100
# def task(mutex):
#     mutex.acquire()
#     global ticket_num
#     temp = ticket_num
#     time.sleep(0.001)
#     ticket_num = temp-1
#     mutex.release()
#
# if __name__ == '__main__':
#     tasklist = []
#     mutex = Lock()
#     for i in  range(100):
#         sub_thread = Thread(target=task,args=(mutex,))
#         sub_thread.start()
#         tasklist.append(sub_thread)
#
#     for task in tasklist:
#         task.join()
#     print("主线程: ",ticket_num)

"""
死锁:
任务拿到之后一直不释放,其他任务就会一直等待对方释放锁的场景
"""

# import time
# from threading import Thread,Lock,current_thread
#
# mutex1 = Lock()
# mutex2 = Lock()
#
# def task():
#     mutex1.acquire()
#     print(f"{current_thread().name}抢到了锁1")
#     mutex2.acquire()
#     print(f"{current_thread().name}抢到了锁2")
#     mutex1.release()
#     mutex2.release()
#
#     mutex2.acquire()
#     print(f"{current_thread().name}抢到了锁2")
#     mutex1.acquire()
#     print(f"{current_thread().name}抢到了锁1")
#     mutex2.release()
#     mutex1.release()
#
# if __name__ == '__main__':
#     for i in range(8):
#         t = Thread(target=task)
#         t.start()



"""
递归锁:(from threading import RLock)
相当于计数器,没acquire一次计数器就会加1,没release一次计数器就会减一
只有计数器为0时,其他进程才能抢到这把锁
只要不出现死锁,其他时候和互斥锁一样
主要应用于递归函数等
"""


"""
信号量
信号量是一个更高级的同步机制，它内部维护一个计数器，用于控制对共享资源的最
大并发访问数量。在 multiprocessing模块中， Semaphore对象用于此类同步。
multiprocessing.Semaphore(value=1) : 创建一个信号量对象， value参数指
定了初始可用的数量，默认为1。
信号量的方法"
acquire([timeout=None]) : 尝试获取信号量。如果信号量可用，则其值减一并
立即返回 True。如果信号量不可用，则阻塞直到超时或信号量变为可用。如果
没有指定 timeout或 timeout为 None，则一直等待直至信号量可用。
release() : 释放一个信号量，其值加一。如果信号量之前已被阻塞，则会唤醒
一个正在等待的进程
"""


