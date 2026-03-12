'''python中的数据类型:
Num,Str,Set,List,Dict,Tuple,Bool
'''
'''
#列表+元组
[
    ("花花",5,'female'),
    ("menglan",10,"male")
]
#列表+字典
[
    {"name":"huahua","age":5,"sex":"famale"},
    {"name":"menglan","age":10,"sex":"male"}
]
# 字典 + 字典
{
    {"name": "huahua", "age": 5, "sex": "famale"},
    {"name": "menglan", "age": 10, "sex": "male"}
}
'''

'''
数据结构有助于:
 管理和利用大型数据集
 从数据库中快速搜索特定数据
 在数据点之间建立清晰的分层或关系连接
 简化并加快数据处理
'''










"""
链表的概念：
•链表是一种线性数据结构，由一系列节点组成，这些节点按照特定的顺序排列，通过指针相互连接
在一起。每个节点包含两部分：元素和指向下一个节点的指针。
"""

# val  next


'''
节点:next
Head:head永远指向第一个节点
Tail:tail节点永远指向最后一个节点
None:链表中的最后一个节点的指针域为None
'''
'''
单链表的操作：
 is_empty() 链表是否为空
 length() 链表长度
 travel() 遍历整个链表
 add(data) 链表头部添加元素
 append(data) 链表尾部添加元素
 insert(pos, data) 指定位置添加元素
 remove(data) 删除节点
 search(data) 查找节点是否存在
'''

