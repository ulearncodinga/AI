# python实现队列

class Queue():
    def __init__(self):
        self.__list = []

    def is_empty(self):
        # 队列是否为空
        return self.__list == []

    def enqueue(self, data):
        # 从队列尾添加一个元素
        self.__list.append(data)

    def dequeue(self):
        # 从队列头移除并返回第一个元素
        return self.__list.pop(0)

    def size(self):
        # 返回队列的大小
        return len(self.__list)

if __name__ == "__main__":
    queue = Queue()
    queue.enqueue(10)
    queue.enqueue(50)
    queue.enqueue(100)  # 队头[10，50，100]队尾
    print(queue.size()) # 3
    print(queue.dequeue())  # 队头[50，100]队尾
    print(queue.size()) # 2
    print(queue.is_empty()) # false








