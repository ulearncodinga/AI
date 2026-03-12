#在Python中,记录日志使用logging库,日志的级别从高到低分别为:
#1.CRITICAL :系统崩溃级别的错误,必须立即处理
#2.ERROR: 运行时的错误,可能会导致程序无法正常进行
#3.WARNING:警告信息
#4.INFO: 信息性信息,程序正常运行
#5.DEBUG:详细信息,通常在诊断问题时有用

# import logging
# logging.basicConfig(level=logging.DEBUG)
# logging.critical('这是一个critical信息')
# logging.error('这时一个error信息')
# logging.warning('这是一个警告信息')
# logging.info('这是一个info信息')
# logging.debug('这是一个debug信息')
# #因为级别问题所以只会打印前三个
# #我们可以在logging.basicConfig中设置级别



#向指定的日志文件里去打印日志信息
# import logging
# logging.basicConfig(filename='./app.log',level=logging.DEBUG,filemode='a',format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')#要是把a改为w新添加的内容就会直接覆盖之前就有的内容
# logging.warning('这是一个警告信息')



#%(name)s:日志记录器的名字
#%(levelname)s:日志级别
#%(asctime)s:时间
#%(message)s:消息本身


#创建自己的日志记录器处理器
import logging
logger = logging.getLogger('mylogger')
logging.basicConfig(filename='./app.log',level=logging.DEBUG,filemode='a',format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
logger.warning('这是我自己定义的日志处理器所记录的日志')