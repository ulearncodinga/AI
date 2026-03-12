# 队列是从对尾插入
# 队列最前面的元素才能取出或者删除

# 将队列允许插入的一端为队尾,将允许删除的一端为队头
'''
在队列中的插入和删除都遵循先进先出（First
in First out，FIFO）的原则。元素可以在任何时刻从队尾插入，但是只有在队列最前面
的元素才能被取出或者删除。通常将队列中允许插入的一端称为队尾，将允许删除的一
端称为队头。队列不允许在中间部位进行操作

队列的操作：
 Queue() 创建一个空的队列
 is_empty() 队列是否为空
 enqueue(data) 从队列尾添加一个元素
 dequeue(data) 从队列头移除并返回第一个元素
 size() 返回队列的大小
'''


class Deque:
    def __init__(self):
        # 初始化空双端队列
        self.items = []

    def is_empty(self):
        # 判断队列是否为空
        return len(self.items) == 0

    def add_front(self, data):
        # 队头插入元素
        self.items.insert(0, data)

    def add_rear(self, data):
        # 队尾插入元素
        self.items.append(data)

    def remove_front(self):
        # 队头删除元素（若队空则抛出提示）
        if self.is_empty():
            raise IndexError("双端队列已空，无法执行队头删除")
        return self.items.pop(0)

    def remove_rear(self):
        # 队尾删除元素（若队空则抛出提示）
        if self.is_empty():
            raise IndexError("双端队列已空，无法执行队尾删除")
        return self.items.pop()

    def size(self):
        # 返回队列长度
        return len(self.items)


# 测试示例
if __name__ == "__main__":
    dq = Deque()
    dq.add_front(10)  # 队头加10，队列：[10]
    dq.add_rear(20)  # 队尾加20，队列：[10,20]
    print(dq.size())  # 输出：2
    print(dq.remove_front())  # 输出：10（队头删除）
    print(dq.remove_rear())  # 输出：20（队尾删除）
    print(dq.is_empty())  # 输出：True