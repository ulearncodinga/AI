"""
记录日志:记录函数调用时间等
"""
import logging
def log_decorator(func):
    def wrapper(*args,**kwargs):
        logging.basicConfig(filename='./app.log',level=logging.INFO,filemode='a',format='%(name)s - %(levelname)s - %(asctime)s - %(message)s')
        logging.warning(f"Calling function:{func.__name__}with args:{args} and kwargs:{kwargs}")
        result = func(*args,**kwargs)
        logging.warning(f"Function{func.__name__}returned:{result}")
        return result
    return wrapper

@log_decorator
def test():
    print('123')
test()