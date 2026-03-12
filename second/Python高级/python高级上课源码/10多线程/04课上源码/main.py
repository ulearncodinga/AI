from concurrent.futures import ThreadPoolExecutor


# 定义一个函数，计算一个数字的平方
def calculate_square(number):
    return number * number


# 数字列表
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 使用with语句创建ThreadPoolExecutor实例
with ThreadPoolExecutor(max_workers=5) as executor:

    results = []
    result = executor.map(calculate_square, numbers)
    print(result)
    print(list(result))
