"""
测试sys.argv的作用
"""
from PyQt5.Qt import *
import sys

#打印列表,第一个元素为脚本名称
print(sys.argv)

#创建QApplication的实例
app = QApplication(sys.argv)
#arguments()返回一个包含所有命令行参数的列表
print(f"arguments:{app.arguments()}")


#qApp,是QApplication的一个全局实例,qApp始终指向正在运行的QApplication的实例,qApp和app都可以来访问命令行的参数
print(f'qApp:{qApp.arguments()}')
