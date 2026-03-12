# Python内置队列模块queue

import queue

q = queue.Queue(5)

q.put(10)
q.put(20)
q.put(30)
q.put(40)
q.put(50)

# 打印队列元素
print(q.queue)
# 打印队列长度
print(q.qsize())
# 队列是否为空
print(q.empty())
# 队列是否满了
print(q.full())
print('-----------------')
data_q = q.get()
print(data_q)
print(q.queue)
print(q.qsize())
print(q.empty())






