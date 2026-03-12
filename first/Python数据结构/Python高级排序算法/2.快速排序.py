'''快速排序：也是一种常见且高效的采用分治策略的排序算法，它通过选取一个“基准”元素，
通过一次排序将待排序的数据分割成独立的两部分，小于基准的放左边，大于基准的放右边。
然后递归地对左右两部分进行排序。
算法步骤：
1. 从未排序的序列中挑出一个元素，称为“基准”（通常选取第一个数据)）。
2. 重新排序数列，所有元素比基准值小的摆放在基准左边，所有元素比基准值大的摆在
基准的右边（相同的数可以到任一边）。在这个分区结束之后，该基准就处于序列的
中间位置。这个称为分区（partition）操作。
3. 递归地把小于基准值元素的子序列和大于基准值元素的子序列排序。'''



# arr = [64,34,25,12,22,11,90]
'''
左边(小于基准)              基准            右边(大于基准值)
left:[34,25,12,22,11]   mid[64]         right[90]
left:[25,12,22,11]      mid[34]         right[]
left:[12,22,11]         mid[25]         right[]
left:[11]               mid[12]         right[22]
'''

def quick_sort(arr):
    #终止递归的条件
    if len(arr) <= 1:
        return arr


    #选择基准值
    pivot = arr[0]
    left = []
    for x in arr:
        if x < pivot:
            left.append(x)

    #列表推导式
    left = [x for x in arr if x < pivot]
    middle = [x for x in arr if x == pivot]
    right = [x for x in arr if x > pivot]


    return quick_sort(left) + middle + quick_sort(right)

#测试
arr = [64,34,25,12,22,11,90]
result = quick_sort(arr)
print(f"快速排序完成之后的列表:{result}")