# Python实现双链表

class Node:
    """创建双链表节点"""

    def __init__(self, data):
        self.data = data
        self.next = None
        self.prev = None


class DoubleLinkList:
    def __init__(self, node=None):
        # __head是私有属性
        self.__head = node

    def is_empty(self):
        # 链表是否为空
        return self.__head == None

    def length(self):
        # 链表长度
        current = self.__head
        count = 0
        while current != None:
            count += 1
            current = current.next
        return count

    def travel(self):
        # 遍历整个链表
        current = self.__head
        while current != None:
            print(current.data, end=" ")
            current = current.next
        print("")

    def add(self, data):
        # 链表头部添加元素,头插法
        new_node = Node(data)
        if self.is_empty():
            self.__head = new_node
        else:
            new_node.next = self.__head
            self.__head = new_node
            new_node.next.prev = new_node

    def append(self, data):
        # 链表尾部添加元素, 尾插法
        new_node = Node(data)
        if self.is_empty():
            self.__head = new_node
        else:
            current = self.__head
            while current.next != None:
                current = current.next
            current.next = new_node
            new_node.prev = current

    def insert(self, pos, data):
        # 指定位置添加元素
        # pos: 元素的下角标
        if pos > (self.length() - 1):
            self.append(data)
        elif pos <= 0:
            self.add(data)
        else:
            new_node = Node(data)
            current = self.__head
            count = 0
            while count < pos:
                count += 1
                current = current.next
            new_node.next = current
            new_node.prev = current.prev
            current.prev.next = new_node
            current.prev = new_node

    def remove(self, data):
        # 删除节点
        current = self.__head
        while current != None:
            if current.data == data:
                # 判断当前节点元素，与要删除的元素是否相等
                if current == self.__head:
                    # 判断当前节点是否是头节点
                    self.__head = current.next
                    if current.next:
                        # 判断链表中头节点是否为唯一一个节点
                        current.next.prev = None
                else:
                    current.prev.next = current.next
                    if current.next:
                        current.next.prev = current.prev
                break
            else:
                current = current.next

    def search(self, data):
        # 查找节点是否存在
        current = self.__head
        while current != None:
            if current.data == data:
                return True
            else:
                current = current.next
        return False


if __name__ == "__main__":
    doublelinklist = DoubleLinkList()

    # 判断链表是否为空
    print(doublelinklist.is_empty()) # True
    # 打印链表长度
    print(doublelinklist.length()) # 0

    doublelinklist.add(10)  # 10
    doublelinklist.add(20)  # 20,10
    doublelinklist.add(30)  # 30,20,10
    doublelinklist.append(100)
    doublelinklist.append(200)  # 30,20,10,100,200
    #
    # doublelinklist.travel()
    # # 判断链表是否为空
    # print(doublelinklist.is_empty())
    # # 打印链表长度
    # print(doublelinklist.length())
    doublelinklist.insert(2, 500)  # 30,20,500,10,100,200
    doublelinklist.travel()
    doublelinklist.insert(-1, 123)  # 123,30,20,500,10,100,200
    doublelinklist.travel()
    doublelinklist.insert(20, 678)  # 123,30,20,500,10,100,200,678
    doublelinklist.travel()
    print(doublelinklist.search(500))
    print(doublelinklist.search(666))
    doublelinklist.remove(10)
    doublelinklist.remove(123)
    doublelinklist.remove(678)
    doublelinklist.travel()
