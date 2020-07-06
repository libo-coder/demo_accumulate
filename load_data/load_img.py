# -*- coding: utf-8 -*-
"""
从文件夹中加载图片
@author: libo
"""
import os

def load_data():
    img_path = []

    #################################### os.walk()  充当目录遍历器 ####################################
    # os.walk() 方法用于通过在目录树中游走输出在目录中的文件名，向上或者向下。充当目录遍历器
    # 返回：三元组(root,dirs,files)
    #   1. root 所指的是当前正在遍历的这个文件夹的本身的地址
    #   2. dirs 是一个 list ，内容是该文件夹中所有的目录的名字(不包括子目录)
    #   3. files 同样是 list , 内容是该文件夹中所有的文件(不包括子目录)
    #################################################################################################
    for root, dirs, files in os.walk("img/original_test"):
        for filename in (x for x in files if x.endswith(('.png', '.jpg', 'tiff'))):
            filepaths = os.path.join(root, filename)
            img_path.append(filename)

    print('Find {} images'.format(len(img_path)))
    return img_path