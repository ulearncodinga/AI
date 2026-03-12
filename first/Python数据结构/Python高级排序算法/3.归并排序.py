'''
归并排序：是一种典型的分治思想的应用，它将一个大问题分解成两个小问题，分别解决这
两个小问题，然后将解决的小问题合并起来，得到大问题的解。
算法步骤：
1. 将数组分解成两个较小的子数组，直到子数组的大小为1。
2. 递归地对子数组进行排序，并将已排序的子数组合并成一个大的有序数组，直到合并
为1个完整的数组
'''

def merge_sort(arr):
    #停止递归的条件
    if len(arr)<=1:
        return arr


    mid = len(arr) // 2
    left = arr[:mid]
    right = arr[mid:]

    left = merge_sort(left)
    right = merge_sort(right)


    return merge(left,right)

def merge(left,right):
    merged = []
    left_index = 0
    right_index = 0

    while left_index < len(left) and right_index < len(right):
        if left[left_index] < right[right_index]:
            merged.append(left[left_index])
            left_index += 1
        else:
            merged.append(right[right_index])
            right_index += 1

    merged.extend(left[left_index:])
    merged.extend(right[right_index:])

    return merged


#测试
arr = [8,9,1,7,2,3,5,4,6,0]
result = merge_sort(arr)
print(f"归并排序后的列表为:{result}")