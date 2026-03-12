class FibIterator:
    def __init__(self,count):
        if not isinstance(count,int) or count < 0:
            raise  ValueError("生成个数必须是大于等于0的整数")

        self.count = count
        self.a = 0
        self.b = 1
        self.current = 0

    def __iter__(self):
        return self
    def __next__(self):
        if self.current>=self.count:
            raise StopIteration
        if self.current == 0:
            result =self.a
        elif self.current == 1:
            result = self.b
        else:
            result=self.a+self.b
            self.a,self.b = self.b,result

        self.current += 1
        return result

if __name__ == '__main__':
    fib_10 = FibIterator(10)
    print("生成是个斐波那契数",list(fib_10))