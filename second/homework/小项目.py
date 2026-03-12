# 实现一个学生管理系统实现增删改查
# OOP
# 装饰器
# 文件I/O
# 正则表达式
# 异常处理
# 多线程

import json
import re
import threading
import time
from datetime import datetime
#定义装饰器,用于记录,记录日志,执行的时间

def log_operation(func):
    def wrapper(self,*args,**kwargs):
        #记录操作的开始时间
        start_time = time.time()

        #执行函数
        result = func(self,*args,**kwargs)


        #记录操作的结束时间
        end_time = time.time()

        #输出耗时时间
        print(f"操作时间:{end_time - start_time:.4f}秒")
        #返回执行结果
        return result
    return wrapper

#定义学生成绩管理系统类
class StudentScoreSystem(object):
    #初始化操作
    def __init__(self,filename="student.json"):
        #创建文件名
        self.filename = filename
        #创建一个空字典,用来存储学生数据{姓名:分数}
        self.students = {}
        #创建线程锁,确保多线程操作的安全性
        self.lock = threading.Lock()
        #启动时,加载以及有的数据
        self.load_data()

    #加载数据,用多线程
    def load_data(self):
        #加载数据的任务函数
        def load_task():
            try:
                #读取,打开文件
                with open(self.filename,'r',encoding='utf-8') as f:
                    self.students = json.load(f)
                #打印加载成功的信息
                print(f"数据已经从{self.filename}加载成功了,共{len(self.students)}条记录")
            except FileNotFoundError:
                print("数据文件不存在,创建新的文件")
            except json.JSONDecodeError:
                print("数据文件格式错误,创建新的文件")

        #创建加载数据的线程
        load_thread = threading.Thread(target=load_task)
        #启动线程
        load_thread.start()
        #线程等待,阻塞子线程
        load_thread.join()

    #定义保存数据,用多线程
    def save_data(self):
        #定义保存数据的任务函数
        def save_task():
            #获取线程锁(确保数据安全)
            with self.lock:
                try:
                    #打开文件,写入数据
                    with open(self.filename,'w',encoding='utf-8') as f:
                        json.dump(self.students,f,ensure_ascii=False)
                    print(f"数据已经保存到{self.filename}")
                except Exception as e:
                    print(f"保存数据时出错了:{e}")
        #创建多线程保存数据
        save_thread = threading.Thread(target=save_task)
        #启动多线程
        save_thread.start()

    #正则表达式,验证姓名
    #验证名字为中文和英文
    def validata_name(self,name):
        pattern = r"^[\u4E00-\u9FA5A-Za-z]{2,10}$"
        # 验证正则表达式
        return re.match(pattern,name) is not None
    #添加学生信息
    @log_operation
    def add_student(self):
        print("添加学生信息")
        #获取用户输入的学生姓名
        name = input('请输入学生姓名').strip()
        #检查学生姓名是否合格
        if not self.validata_name(name):
            print("姓名格式错误,请使用中文或者英文(2-10)个字符")
            return
        #检查这个学生是否存在
        if name in self.students:
            print(f"学生{name}已经存在,不需要重复添加")
        else:
            try:
                score = float(input("请输入学生的分数:"))
                if score < 0 or score > 100:
                    print("分数必须在0-100之间")
                    return
                #将学生的信息添加到字典中
                self.students[name] = score
                print(f"学生{name},成绩{score}添加成功")
                #保存数据
                self.save_data()
            except ValueError:
                print("输入的分数必须是一个数值")

    #查询学生
    @log_operation
    def query_student(self):
        print("-----查询学生信息-----")
        name = input("请输入你要查询的学生姓名").strip()
        #利用正则表达式验证输入是否合理
        if not self.validata_name(name):
            print("姓名格式错误,重新输入")
            return
        #检查学生是否存在
        if name in self.students:
            print(f"学生{name}的成绩是{self.students[name]}")
        else:
            print(f"系统找不到学生{name}")
    #修改学生信息
    @log_operation
    def update_student(self):
        print("修改学生信息")
        # 获取用户输入的学生姓名
        name = input('请输入你要修改的学生姓名').strip()
        # 检查学生姓名是否合格
        if not self.validata_name(name):
            print("姓名格式错误,请使用中文或者英文(2-10)个字符")
            return
            # 检查这个学生是否存在
        if name in self.students:
             try:
                 #获取一个新的分数
                new_score = float(input("请输入你要修改的学生的分数"))
                if new_score < 0 or new_score > 100:
                    print("分数必须在0-100之间")
                    return
                else:
                    self.students[name] = new_score
                    print(f"学生{name}的成绩已更新,分数为{new_score}")
             except ValueError:
                 print("分数必须是个数值")
        else:
            print(f"系统找不到学生{name}")


    @log_operation
    #删除学生信息
    def delete_student(self):
        print("删除学生信息")
        name = input("请输入你要删除的学生信息").strip()
        if not self.validata_name(name):
            print("姓名格式错误,请使用中文或者英文(2-10)字符")
            return
        if name in self.students:
            del self.students[name]
            #提示信息
            print(f"学生{name}已经删除成功")
            self.save_data()
        else:
            print(f"系统找不到学生{name}")
    #查询所有学生信息
    @log_operation
    def show_all_student(self):
        print("查询所有学生信息")
        if not self.students:
            print("没有学生信息存在")
            return
        for name,score in self.students.items():
            print(f"{name}:{score}")
        print(f"一共有{len(self.students)}名学生")


    def run(self):
        while True:
            print("-------学生信息管理系统--------")
            print("输入1 -> 添加学生信息")
            print("输入2 -> 查询学生信息")
            print("输入3 -> 修改学生信息")
            print("输入4 -> 删除学生信息")
            print("输入5 -> 查询所有学生学生信息")
            print("输入6 -> 推出管理系统")
            try:
                #获取用户输入选项
                choices = input("请输入 1-6 选择的选项")
                if choices == '1':
                    self.add_student()
                elif choices == '2':
                    self.query_student()
                elif choices == '3':
                    self.update_student()
                elif choices == '4':
                    self.delete_student()
                elif choices == '5':
                    self.show_all_student()
                elif choices == '6':
                    print("退出管理系统")
                    self.save_data()
                    time.sleep(0.5)
                    break
                else:
                    print('请输入1-6之间的数字')
            except Exception as e:
                print(f"输入有误,请重新输出{e}")


if __name__ == '__main__':
    #创建实例化对象
    system = StudentScoreSystem()
    #启动
    system.run()
    #程序结束
    print("程序结束")
