# 栈的实现
class Stack:
    '''栈'''
    def __init__(self):# list ,链表
        #构造容器
        self.__list = []


    def push(self,data):
        '''将data压入栈顶'''
        self.__list.append(data)

    def pop(self):
        #将栈顶数据移除并返回
        if not self.is_empty():
            #判断栈是否是空栈
            return self.__list.pop()
        else:
            return "这是一个空栈!"




    def peek(self):
        #查看栈顶元素
        if self.__list:
            return self.__list[-1]
        else:
            return None
    def is_empty(self):
        #检查栈是否为空
        return self.__list == []

    def size(self):
        #获取栈的大小
        return len(self.__list)


if __name__ == '__main__':
    S = Stack()
    S.push(10)
    S.push(20)
    S.push(30)
    S.push(40)   # 栈底 [10,20,30,40] 栈顶

    print(S.size())# 4
    print(S.peek())# 40

    print(S.size())# 4
    print(S.pop())# 40
    print(S.size())# 3
    print(S.is_empty())# False




