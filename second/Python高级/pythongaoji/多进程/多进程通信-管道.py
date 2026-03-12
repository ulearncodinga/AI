'''进程间通信
是指在不同进程之间传送数据或信号的一些方法,在多进程程序中,ipc是非常重要的
因为它允许进程协同工作,共享数据或同步操作,以下是几种常见的进程间通信机制
'''
'''
管道(pipe)
管道是一种在父子进程或兄弟进程间进行通信的机制.Python的multiprocessing模块提供了Pipe()函数
可以用来创建管道

parent_conn,child_conn = Pipe()

使用multiprocessing.Pipe()可以创建一个管道.这个函数返回一个由两个连接对象组成的元组,这两个对象分别代表管道的两端
.默认情况下,管道是双向的,每个端点都可以既读又写

from multiprocessing import Process,Pipe
#创建一个管道
parent_conn,child_conn = Pipe()

'''
'''
在上面的例子中，parent_conn和child_conn都是管道对象，它们都拥有共同的方
法：
send(obj)：发送一个对象到管道的另一端。这个对象必须是可序列化的。
recv()：从管道的另一端接收一个对象。该方法是阻塞的。
close()：关闭管道连接。当不再需要管道时，应该调用这个方法来释放资源。
fileno()：返回由连接对象使用的文件描述符。
poll(timeout)：返回连接对象中是否有可以读取的数据。如果未指定timeout，
会马上返回，如果timeout是一个数字，则指定了阻塞的最大秒数，如果未指定
timeout，那么将一直等待。
send_bytes(buffer, offset, size)：通过连接发送buffer，offset是buffer中的
偏移量，size是要发送的字节数。数据以一条完整的数据发送。
recv_bytes(maxlength)：以字符串的形式返回一条从连接对象另一端发送过来
的字节数据。此方法在接收到数据前一直阻塞。如果连接对象被关闭或没有数据
可读取，将抛出异常。如果消息长度大于maxlength，则会抛出异常且该连接对
象不可再读。
recv_bytes_into(buffer, offset)：将一条完整的数据读入buffer中并返回消息
的字节数，此方法在接收到数据前一直阻塞。如果连接对象被关闭或没有数据可
读取，将抛出异常。offset指定buffer中放置消息处的字节偏移量。如果消息长
度大于buffer将抛出异常。
'''
#在进程中使用管道
from multiprocessing import Process, Pipe
def child_process(conn):
 # 子进程从连接中接收数据
 conn.send('Hello from child')
 data = conn.recv()
 print('Received:', data)
 conn.close()
if __name__ == '__main__':
 # 创建一个管道
 parent_conn, child_conn = Pipe()
 # 创建子进程
 p = Process(target=child_process, args=(child_conn,))
 p.start()
 # 父进程发送数据到子进程
 parent_conn.send('Hello from parent')
 print('Received:', parent_conn.recv())
 # 等待子进程结束
 p.join()

"""
管道特点
双向通信：管道允许两个方向的通信，即每个管道有一个接收端和一个发送端。
点对点连接：管道通常用于两个进程之间的直接通信，不支持多个进程之间的通
信。
管道大小有限：管道的缓冲区大小是有限的。如果缓冲区满了，发送操作会阻
塞。
注意事项
管道默认是双向的，但也可以通过设置 duplex=False来创建单向管道。此时返
回的第一个对象只能接收消息，第二个对象只能发送消息。
当使用管道在进程间传递大量数据时，要注意管道可能会成为性能瓶颈。
"""
# 单向管道
from multiprocessing import Process, Pipe
def sender(conn):
 conn.send([42, None, 'hello'])
 conn.close()
if __name__ == '__main__':
 # 创建单向管道，parten_conn是只读的，child_conn是只写的
 parent_conn, child_conn = Pipe(duplex=False)
 # 创建发送者进程
 # sender函数应该使用child_conn，因为它是可写的
 sender_process = Process(target=sender, args=(child_conn,))
 # 启动进程
 sender_process.start()
 print(parent_conn.recv())
 # 等待进程结束
 sender_process.join()
