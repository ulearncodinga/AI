'''
消息队列
消息队列提供了一种在进程间传输数据的方式,这种方式是通过在内核中维护一个消息队列来实现的.
进程可以发送数据到队列.在Python的multiprocessing模块中,Queue类提供了一个先进先出(FIFO)的消息队列


栈:先进后出


'''
'''
from multiprocessing import Queue
#创建一个消息队列
queue = Queue(maxsize=10)#maxsize为队列中最多可以存放的元素数量
'''
#消息队列的方法
'''
在上面的例子中，queue是创建出来的消息队列，它拥有下面几种方法：
put(obj, block=True, timeout=None)：将obj放入队列，如果可选参数
block是True而且timeout是None，将会阻塞当前进程，直到有空的缓冲槽。如
果timeout是正数，将会在阻塞了最多timeout秒之后还是没有可用的缓冲槽时抛
出异常。如果block是False，那么在没有空的缓冲槽时，会立即抛出异常，此时
timeout会被忽略。(发送数据)
get(block=True, timeout=None)：从消息队列里获取消息。该方法为阻塞等
待的方法。block和timeout的作用与put一致。(接收数据)
empty()：如果队列为空，返回 True，否则返回 False。
full()：如果队列满了，返回 True，否则返回 False。
qsize() : 返回队列中当前元素的数量。
get_nowait()：立即尝试从队列里获取一个元素，如果队列为空，抛出
Queue.Empty异常。
put_nowait()：立即尝试向队列里放入一个元素，如果队列满了，抛出
Queue.Full异常。
'''
from multiprocessing import Process,Queue
import time

def process1(process_queue):
    print('准备接收数据')
    print('接收到的数据为:',process_queue.get())

if __name__ == '__main__':
    process_queue = Queue(5)
    p1 = Process(target=process1,args=(process_queue,))
    p1.start()
    time.sleep(1)
    process_queue.put('Hello')
    p1.join()
    p1.close()

#消息队列的特点
'''
1.先进先出(FIFO):队列遵循先进先出(FIFO)的原则,即先放入队列的元素会先被取出
2.同步访问:Queue类提供了一系列同步方法,如put(),get()等,以确保多进程对队列的访问是安全的
3.容量限制:队列可以指定最大容量,当对列满时,新元素将无法放入;当对列空时,试图从队列中获取元素的进程将会被阻塞,直到有新元素放入队列
4.生产者消费者模式:Queue类非常适合用于生产者-消费者模式,其中生产者进程将数据放入队列,而消费者进程从队列中取出数据
'''
from multiprocessing import Process,Queue
import time
#生产者函数
def producer(queue):
    for i in range(5):
        queue.put(f"Product {i}")
        print(f"Produced {i}")
        time.sleep(1)
#消费者函数
def consumer(queue):
    while True:
        product = queue.get()
        if product is None:
            break
        print(f"Cousumed {product}")

if __name__ == '__main__':
    queue = Queue(5)
    p = Process(target=producer,args=(queue,))
    c = Process(target=consumer,args=(queue,))

    p.start()
    c.start()
    p.join()
    queue.put(None)
    c.join()
#注意事项
'''
避免全局共享:不要在多个进程间共享同一个队列实例,而应该为每个进程创建的单独的队列实例
设置队列大小:进入过没有指定队列的大小,他将默认为无限大小,这可能导致内存问题,
特别是当生产者产生消息的速度远大于消费者消费消息的速度时
处理队列异常:当队列操作失败时(如队列已满或为空),应该去捕获并且处理响应的异常
队列的性能:队列操作可能会影响性能,尤其是高并发环境下,需要根据应用需求选择合适的队列实现和大小
'''


#进程间通信(IPC):管道 共享内存 信号 队列
from multiprocessing import Process,Queue
import time

def task1(q):
    for i in ["葱爆牛肉","佛跳墙","宫保鸡丁"]:
        time.sleep(2)
        q.put(i)
        print(f"task1:{i} 已经做好")

def task2(q):
    while True:
        print("task2:",q.get())
        time.sleep(2)

if __name__ == '__main__':
    q = Queue()
    task1_process = Process(target=task1,args=(q,))
    task2_process = Process(target=task2,args=(q,))
    task1_process.start()
    task2_process.start()
    task1_process.join()
    task2_process.join()
    task1_process.close()
    task2_process.close()












