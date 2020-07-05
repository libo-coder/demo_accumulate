# -*- coding: utf-8 -*-
"""
函数块运行时间统计：在函数块前加 @timer 即可
"""
import time

def timer(func):
    def new_func(*args, **args2):
        t0 = time.time()
        back = func(*args, **args2)
        print("%.3fs taken for {%s}" % (time.time()-t0, func.__name__))
        return back
    return new_func()


