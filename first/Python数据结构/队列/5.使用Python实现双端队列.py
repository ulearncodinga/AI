# Python实现双端队列

class Deque:
    """双端队列"""
    def __init__(self):
        self.__list = []


    def add_front(self,data):
        #从队头加入一个元素
        self.__list.insert(0,data)
    def add_rear(self,data):
        #    从队尾加入一个元素
        self.__list.append(data)
    def remove_front(self):
        #从队头删除一个元素
        return self.__list.pop(0)


    def remove_rear(self):
        #从队尾删除一个元素
        return self.__list.pop()

    def is_empty(self):
        #判断队列是否为空
        return  self.__list == []

    def size(self):
        #返回队列的大小
        return len(self.__list)

if __name__ == '__main__':
    deque = Deque()
    deque.add_front(4)
    deque.add_front(3)
    deque.add_front(2)
    deque.add_front(1)
    deque.add_rear(5)
    print(deque.remove_front())
    print(deque.size())
    print(deque.remove_rear())
    print(deque.size())
