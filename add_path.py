# -*- coding: utf-8 -*-
"""
OCR 数据读入添加数据的绝对路径
@author: libo
"""

labelPath = 'D:/work_xinhuo/txt_add_path/txt/test.txt'
imgPath = 'D:/work_xinhuo/txt_add_path/img/'
labelPath2 = 'D:/work_xinhuo/txt_add_path/txt/test_add.txt'

with open(labelPath, 'r', encoding='utf-8') as f:
    file_names = f.readlines()
print('-----------------', file_names[0:2])      # 左闭右开

file_names2 = []
for strName in file_names:
    strName2 = imgPath + strName
    file_names2.append(strName2)

with open(labelPath2, 'w') as w:
    w.writelines(file_names2)
print('=================', file_names2)