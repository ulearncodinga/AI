# Python实现哈希表

# 创建一个空字典
my_dict = {}

# 插入数据（键值对）
my_dict["huahua"] = 1
my_dict["heye"] = 2
my_dict["menglan"] = 3

# 查找值
print(my_dict["heye"])

# 修改值
my_dict["heye"] = 4
print(my_dict["heye"])

# 删除键值对
del my_dict["heye"]
print(my_dict)

# 添加键值对
my_dict["xiaoming"] = 5
print(my_dict)
