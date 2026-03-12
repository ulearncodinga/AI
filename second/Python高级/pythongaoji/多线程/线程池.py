#线程池如何返回结果
import time
from concurrent.futures import ThreadPoolExecutor
pool = ThreadPoolExecutor(10)
def task(name):
    print(name)
    time.sleep(3)
    return name+10
if __name__ == '__main__':
    future_ls = []
    for i in range(50):
        future = pool.submit(task,i)
        future_ls.append(future)
    pool.shutdown()
    for future in future_ls:
        print("函数返回的结果: ",future.result())