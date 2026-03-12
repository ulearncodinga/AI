# Python 实现栈

class Stack:
    """栈"""

    def __init__(self):
        """构造容器，不希望外部可以操作这个列表，所以构造私有属性"""
        self.__list = []

    def push(self, data):
        """添加一个新元素，压入栈顶"""
        self.__list.append(data)
 
    def pop(self):
        """弹出一个栈顶的元素（移除且返回）"""
        if not self.is_empty():  # 添加一个检查，确保栈不为空
            return self.__list.pop()
        else:
            return "这是一个空栈！"  # 如果栈为空，则抛出异常

    def peek(self):
        """返回栈顶元素，不移除元素"""
        if self.__list:
            return self.__list[-1]
        else:
            return None

    def is_empty(self):
        """判断栈是否为空"""
        return self.__list == []       # 遇到return之后，会先判断return右侧的句子，判断后，返回句子的指针

    def size(self):
        """返回栈的元素个数，不返回值"""
        return len(self.__list)


if __name__ == "__main__":
    stack = Stack()
    stack.push(0)
    stack.push(1)
    stack.push(2)
    stack.push(3)
    print(f'栈的长度：{stack.size()}')
    print(f'查看栈顶元素：{stack.peek()}')
    print(f'栈的长度：{stack.size()}')
    print(f'弹出栈顶元素：{stack.pop()}')
    print(f'栈的长度：{stack.size()}')
    print(f'判断是否为空栈：{stack.is_empty()}')
