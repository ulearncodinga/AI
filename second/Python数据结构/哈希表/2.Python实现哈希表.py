#python实现哈希表


#创建一个空字典

my_dict = {}


#插入数据(键值对)
my_dict["花花"] = 1
my_dict["Carl"] =2
my_dict['Mike'] = 3

#查找值
print(my_dict["Carl"])

#修改值
my_dict["Mike"] = 4
print(my_dict["Mike"])

#删除键值对
del my_dict["Carl"]
print(my_dict)

#添加键值对
my_dict["Liu"] = 5
print(my_dict)