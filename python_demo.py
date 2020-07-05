# -*- coding: utf-8 -*-
"""
一些常用的 Python 代码合计整理
@author: libo
"""

def all_unique(lst):
    """ 检查给定的列表是不是存在重复元素  使用 set() 函数来移除所有重复元素 """
    return len(lst) == len(set(lst))


def most_frequent(lst):
    """ 根据元素频率取列表中最常见的元素 """
    return max(set(lst), key=lst.count)


from collections import Counter
def anagram(first, second):
    """ 检查两个字符串的组成元素是不是一样的 """
    return Counter(first) == Counter(second)


def byte_size(string):
    """ 检查字符串占用的字节数 """
    return len(string.encode('utf-8'))


from math import ceil
def chunk(lst, size):
    """ 分块：给定具体的大小，定义一个函数以按照这个大小切割列表 """
    return list(map(lambda x: lst[x * size, x * size + size], list(range(0, ceil(len(lst) / size)))))


def compact(lst):
    """ 压缩：将布尔值过滤掉  使用 filter() 函数 """
    return list(filter(bool, lst))


def unzip(array):
    """ 解包：将打包好的成对列表解开成两组不同的元组 """
    transposed = zip(*array)
    return transposed

# array = [['a', 'b'], ['c', 'd'], ['e', 'f']]
# print(unzip(array))         # [('a', 'c', 'e'), ('b', 'd', 'e')]


def spread(arg):
    ret = []
    for i in arg:
        if isinstance(i, list):
            ret.extend(i)
        else:
            ret.append(i)
    return ret

def deep_flatten(lst):
    """ 通过递归的方式将列表的嵌套展开为单个列表 """
    result = []
    result.extend(spread(list(map(lambda x: deep_flatten(x) if type(x) == list else x, lst))))
    return result

# deep_flatten([1, [2], [[3], 4], 5])     # [1, 2, 3, 4, 5]


def different(a, b):
    """ 列表的差：返回第一个列表的元素，其不在第二个列表内 """
    set_a = set(a)
    set_b = set(b)
    comparison = set_a.difference(set_b)
    return comparison


def to_dictionary(keys, values):
    """ 将两个列表转化为字典 """
    return dict(zip(keys, values))


def palindrome(string):
    """ 回文序列 """
    from re import sub
    s = sub('[\W_]', '', string.lower())
    return s == s[::-1]