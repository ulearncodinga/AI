# # os模块
# import os
# # 使用os模块创建一个文件夹
#
# # 定义一个变量，用来代表要创建的文件夹的名字
# folder_name = 'NEWfolder'
#
# # 用来检查你要创建的文件夹是否存在
# if not os.path.exists(folder_name):
#     # 当文件夹不存在时，在if分支里创建该文件夹
#     os.mkdir(folder_name)
#     print(f'{folder_name}已创建成功！！！')
# else:
#     print('该文件夹已存在！！')




# # sys模块
# import sys
#
# # 使用sys模块打印Python解释器版本
# print(f'该文件使用的Python解释器的版本为：', sys.version)
#
# # 终止程序
# sys.exit(0)
#
# print('123')



# math模块
# import math
# print(math.pi)
#
# a = math.pi
# print(math.sin(a))


# # time模块
# import time
# start = time.time()
# time.sleep(1)
# stop = time.time()
# print(stop - start)

# random模块
import random
# a = random.randint(1, 10)
# print(a)


my_list = [1, 2, 3, 4, 5]
# 将列表中的元素随机打乱
random.shuffle(my_list)
print(my_list)


