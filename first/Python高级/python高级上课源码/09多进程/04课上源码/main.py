from concurrent.futures import ProcessPoolExecutor
import time


# 定义一个简单的函数，用于执行一个计算密集型的任务
def compute_square(n):
    return n * n


if __name__ == '__main__':
    numbers = [1, 2, 3, 4, 5]
    results = []

    # 使用with语句创建ProcessPoolExecutor实例
    with ProcessPoolExecutor(max_workers=3) as executor:

        # 提交计算任务到进程池
        future = executor.map(compute_square, numbers)
        for res in future:
            print(res)

