# 线性查找,在列表中循环,与需要查找的目标值对比

# def linear_search(num_list,target_num):
#     for i in range(len(num_list)):
#         if num_list[i] == target_num:
#             return i
#     return  -1
#
#
# num_list = [1,2,3,4,5,5,6,7,8,9,10]
# target_num = 5
# result = linear_search(num_list,target_num)
# if result != -1:
#     print(f"找到目标值{target_num}在索引{result}")
# else:
#     print(f"没有找到目标值!")






#二分查找

def linbary_search(num_list,target_num):
    num_list.sort()# sort()将列表进行排序,从小到大顺序,会改变袁术列表,sorted函数不会改变原始列表

    #索引值
    low = 0
    hight = len(num_list) - 1


    while low <= hight:
        mid = (low + hight) // 2
        mid_num = num_list[mid]
        if mid_num == target_num:
            return mid
        elif mid_num > target_num:
            hight = mid - 1

        else:
            low = mid + 1
    return None

num_list = [4,9,7,5,6,2,3,1,10,540,54650,45,64]
target_num = 10
result = linbary_search(num_list,target_num)
print(f"排序后的列表:{num_list}")
if result != None:
    print(f"二分查找找到目标值:{target_num}在索引{result}")
else:
    print("没有该元素!")





























