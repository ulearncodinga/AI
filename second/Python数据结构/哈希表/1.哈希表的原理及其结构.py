'''哈希表（Hash），也可以叫做散列表。哈希表是一种特殊的数据结构，它与数组、链表
以及我们之前学到过的数据结构相比，有很明显的区别。它可以提供快速插入和查找的
操作，不论哈希表中有多少数据，我们对其进行操作时，平均时间复杂度接近于O(1)，也
就是常数时间。
哈希表的原理：它利用哈希函数将给定的键（key）映射到数据的存储位置。通过哈希函
数可以快速地插入、查找、删除数据。'''
'''
哈希函数：是用于计算对象的哈希值的一种内置机制，它对于支持哈希操作（如字典 
dict 的key）的对象非常重要。哈希函数的目标是将任意大小、复杂度各异的数据（如字
符串、整数、自定义对象等）转换为固定长度的整数，这个整数称为哈希值（或散列
值）。哈希值主要用于快速查找、比较和索引数据，特别是在实现哈希表这样的数据结
构时。
注意：可变的对象通常是不可哈希的，因为他们的值可以改变，这会导致哈希值不一致。
'''
'''
哈希冲突：由于哈希表大小有限，而需要我们存储的数据总量再增大，总会发生上述案
例的冲突，即不同key值通过哈希函数后会产生相同的地址，一般来说，哈希冲突是无法
避免的，所以我们计算机科学家，就提出了几种解决办法：
1. 开放寻址法：如果发生了哈希冲突，则可以向后探查新的位置来存储这个值。
2. 链地址法（拉链法）：哈希表的每个位置都链接一个链表，当冲突发生的时候，冲突
的元素将被加到该位置的链表最后
'''

'''
在Python中有一个内置的数据结构，可以实现哈希表的功能，它就是字典（dictionary）。
Python字典（dict）是一种无序的，可变的集合，他的元素以“键值对（key-value）”的
形式存储。字典中的key是唯一的且不可变的，这意味着它们一旦设置就无法更改。
在底层，Python的字典以哈希表的形式运行，当我们创建字典并添加键值对时，Python
会将哈希函数作用于键，从而生成哈希值，接着哈希值决定对应的值将存储在内存的哪
个位置中，所以当我们检索的时候，Python就会对键进行哈希，从而快速引导Python找
到值的位置，而无需考虑字典的大小。
'''

# keys    Hash Function    index    values

class HashTable:
    def __init__(self, size=10):
        self.size = size
        self.table = [[] for _ in range(size)]  # 链地址法初始化

    def hash_func(self, key):
        return hash(key) % self.size  # 哈希函数：取模定索引

    def add(self, key, value):
        index = self.hash_func(key)
        for item in self.table[index]:
            if item[0] == key:
                item[1] = value  # 键存在则更新值
                return
        self.table[index].append([key, value])  # 键不存在则新增

    def get(self, key):
        index = self.hash_func(key)
        for item in self.table[index]:
            if item[0] == key:
                return item[1]  # 找到键返回对应值
        return None  # 键不存在返回None

    def remove(self, key):
        index = self.hash_func(key)
        for i, item in enumerate(self.table[index]):
            if item[0] == key:
                del self.table[index][i]  # 删除对应键值对
                return
# 创建哈希表实例
ht = HashTable(size=5)

# 添加键值对
ht.add("name", "Alice")
ht.add("age", 30)

# 获取并打印值
print(ht.get("name"))  # 输出：Alice
print(ht.get("age"))   # 输出：30

# 删除键值对后再获取
ht.remove("age")
print(ht.get("age"))   # 输出：None

