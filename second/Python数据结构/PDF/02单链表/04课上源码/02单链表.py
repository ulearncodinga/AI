class Node:
    """创建单链表节点"""

    def __init__(self, data):
        self.data = data
        self.next = None


class LinkList:
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
        new_node.next = self.__head
        self.__head = new_node

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

    def insert(self, pos, data):
        # 指定位置添加元素
        # pos: 元素的下角标
        if pos > (self.length() - 1):
            self.append(data)
        elif pos <= 0:
            self.add(data)
        else:
            new_node = Node(data)
            pre = self.__head
            count = 0
            while count < (pos - 1):
                count += 1
                pre = pre.next
            new_node.next = pre.next
            pre.next = new_node

    def remove(self, data):
        # 删除节点
        current = self.__head
        pre = None
        while current != None:
            if current.data == data:
                # 判断当前节点元素，与要删除的元素是否相等
                if current == self.__head:
                    self.__head = current.next
                else:
                    pre.next = current.next
                break
            else:
                pre = current
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
    linklist = LinkList()

    # 判断链表是否为空
    print(linklist.is_empty())
    # 打印链表长度
    print(linklist.length())

    linklist.add(10)  # 10
    linklist.add(20)  # 20,10
    linklist.add(30)  # 30,20,10
    linklist.append(100)
    linklist.append(200)  # 30,20,10,100,200

    # linklist.travel()
    # # 判断链表是否为空
    # print(linklist.is_empty())
    # # 打印链表长度
    # print(linklist.length())
    linklist.insert(2, 500)  # 30,20,500,10,100,200
    linklist.travel()
    linklist.insert(-1, 123)  # 123,30,20,500,10,100,200
    linklist.travel()
    linklist.insert(20, 678)  # 123,30,20,500,10,100,200,678
    linklist.travel()
    print(linklist.search(500))
    print(linklist.search(666))
