'''
进程池是一组预先创建的空闲进程，它们等待执行任务，主进程负责将任务分配给进
程池中的空闲进程去执行。进程池可以管理进程的创建和销毁，避免了频繁地创建和
销毁进程带来的开销，通过进程池可以轻松的实现多任务的并行处理。
为什么使用进程池
效率：相比于手动创建和管理多个进程，使用进程池可以更高效地利用系统资
源。
简化：进程池简化了并行编程的复杂性，开发者不需要关注进程的创建和销毁细
节。
控制：可以限制同时运行的进程数量，防止系统资源被过度消耗。
进程池的创建
在Python中，进程池的创建有两种方式。
1.使用multiprocessing库

import multiprocessing.Pool

multiprocessing.Pool(processes=None, initializer=None, initargs=
(), maxtasksperchild=None)

processes：进程池中的进程数。如果 processes为 None，则默认使用系统的
处理器核心数。
initializer：每个工作进程启动时要执行的可调用对象，默认为None。如果是
None，则调用initializer(*initargs)。
initargs：传递给 initializer的可变参数元组。
maxtasksperchild：工作进程退出之前可以完成的任务数，完成后用一个新的
工作进程来替代原进程，来让闲置的资源被释放。maxtasksperchild默认是
None，意味着只要Pool存在工作进程就会一直存活
**方法**
apply(func, args=())：在一个池工作进程中执行func(args,*kwargs),然后返回
结果。需要强调的是:此操作并不会在所有池工作进程中并执行func函数。如果要
通过不同参数并发地执行func函数，必须从不同线程调用p.apply()函数或者使用
p.apply_async。它是阻塞的。apply很少使用。
apply_async(func, args=(), kwds={}, callback=None)：异步地执行函数
func。 args是传递给 func的位置参数元组， kwds是传递给 func的关键字参
数字典。callback是一个回调函数，当 func执行完成后会被调用。
map(func, iterable, chunksize=None)：将 iterable中的每个元素作为参数
传递给func，并返回结果列表。此方法类似于内置函数 map，但是它是并行的。
map_async(func, iterable, chunksize=None, callback=None)：与 map类
似，但是是异步的。chunksize指定每次分配给进程的迭代器元素数量。
imap(func, iterable, chunksize)：imap 与 map的区别是，map是当所有的
进程都已经执行完了，再返回结果，imap()则是立即返回一个iterable可迭代对
象。
imap_unordered(func, iterable, chunksize)：与imap相同，只不过不保证返
回的结果顺序与进程添加的顺序一致。
close()：阻止任何新的任务被提交到池中。一旦所有任务完成，工作进程会退
出。
terminate()：立即终止所有工作进程，不再处理未处理的任务。
join()：等待所有工作进程退出。必须在 close()或 terminate()之后调用。
2.
'''
# import time
# from concurrent.futures import ProcessPoolExecutor
#
# pool = ProcessPoolExecutor(10)
# def task(name):
#     print(name)
#     time.sleep(3)
# if __name__ == '__main__':
#     for i in range(50):
#         pool.submit(task,i)


#apply的使用

# import time
# from multiprocessing import Pool
#
# def factorial(n):
#     while True:
#         if n<=1:
#            break
#         print(n)
#         n -= 1
#         time.sleep(1)
# def test1():
#     print(123)
# def test2():
#     print(456)
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         result = pool.apply(factorial,(5,))
#         # print(result)
#         pool.apply(test1)
#         pool.apply(test2)




#apply_async的使用

# from multiprocessing import Pool
# def fbnc(n):
#     if n<=1:
#         return n
#     else:
#         return fbnc(n-1) + fbnc(n-2)
# def handle_result(result):
#     print(f'Fibonacci result:{result}')
#
# if __name__ == '__main__':
#     with Pool(processes=4) as pool:
#         fib_indices = [35,36,37,38]
#         results = []
#         for index in fib_indices:
#             result = pool.apply_async(fbnc,(index,),callback=handle_result)
#             results.append(result)
#         pool.close()
#         pool.join()
#     print("ALL Fibonacci calculations are complete")

"""
使用concurrent.futures库的ProcessPoolExecutor创建进程池
concurrent.futures.ProcessPoolExecutor(max_workers=None, 
mp_context=None,initializer=None, initargs=())

max_workers : 指定进程池中可以同时运行的最大进程数。如果设置为 None 或
未指定，则默认为机器的处理器数量，最多为61。
mp_context : 指定多进程上下文。默认情况下， ProcessPoolExecutor 使用
multiprocessing.get_context() 来获取上下文。这允许你选择不同的上下
文，例如 spawn、 fork、 forkserver 等，这些上下文可能提供不同的功能，
如更好的资源隔离、更好的安全性等。
initializer : 一个可选的可调用对象，每个工作进程在启动时都会调用它。这
可以用来执行进程的初始化操作，例如设置进程局部存储。
initargs : 一个元组，其中包含传递给 initializer 的参数。

主要方法:
submit(fn, *args, **kwargs) : 提交一个可调用对象 fn 到进程池，并返回一
个 Future 对象，该对象的result方法可以用来获取结果。
map(func, *iterables, timeout=None, chunksize=1) : 它允许你将一个函
数 func 应用于多个可迭代对象 iterables 中的元素，并且并行地在多个进程
上执行这些函数调用。timeout是可选参数，用于设置阻塞等待每个任务完成的
最大秒数。如果 timeout 被触发，将引发
concurrent.futures.TimeoutError。 chunksize是可选参数，指定每次提交
给进程池的任务数量。一个较大的 chunksize 可以减少进程间通信的开销，但
它也会增加内存消耗，因为它会保存更多的任务结果。返回的结果是一个迭代
器。
shutdown(wait=True) : 等待所有进程完成当前任务后关闭进程池。如果 wait
参数设置为True，进程池会等待所有任务完成；如果设置为 False，进程池会
立即返回，不等待任务完成。使用with语句管理时，with语句结束时会自动调用
shutdown


"""



"""
 手动创建多进程
优点：
1. 灵活性：手动创建进程可以更精确地控制每个进程的创建和销毁，可以针对特定
任务定制进程的行为。
2. 直接控制：可以直接与进程对象交互，例如，可以设置进程的名称、守护状态
等。
3. 简单的并行结构：对于简单的并行任务，手动创建进程可以更加直观。
缺点：
1. 资源管理：需要手动管理进程的生命周期，包括创建、销毁和异常处理。
2. 开销大：进程的创建和销毁开销较大，如果频繁创建和销毁大量进程，可能会导
致性能问题。
3. 同步问题：需要手动处理进程间同步（例如，使用锁、信号量等）。
进程池创建多进程
优点：
1. 高效管理：进程池会管理进程的生命周期，包括进程的创建和销毁，减少了开
销。
2. 资源限制：可以限制同时运行的进程数量，防止系统资源被过度消耗。
3. 简化并行：简化了并行任务的处理，不需要手动创建和销毁进程。
4. 任务分发：可以使用 apply , apply_async , map , map_async等函数方便地分发
任务到进程池。
缺点：
1. 灵活性较低：与手动创建进程相比，进程池提供的是一种更通用的解决方案，可
能不适合所有特定场景。
2. 可能造成阻塞：如果进程池中的所有进程都在忙，而又有新的任务需要执行，那
么这些任务可能会被阻塞，直到有进程可用。
"""


















