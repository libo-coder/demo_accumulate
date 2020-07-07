# -*- coding: utf-8 -*-
"""
训练日志处理
@author: libo
"""
import re
import matplotlib.pyplot as plt
import numpy as np

def main():
    file = open('log/log_train.txt', 'r')

    # loss1 = []
    # for line in file:
    #     m = re.search('Train loss', line)
    #     if m:
    #         n = re.search('[0-9]\.[0-9]+', line)  # 正则表达式
    #         if n is not None:
    #             loss1.append(float(n.group()))    # 提取精度数字
    #             print("loss1: ", loss1)


    # loss2 = []
    # # search the line including accuracy
    # for line in file:
    #     m = re.search('Valid loss', line)
    #     if m:
    #         n = re.search('[0-9]\.[0-9]+', line)      # 正则表达式
    #         if n is not None:
    #             loss2.append(float(n.group()))        # 提取精度数字
    #             print("loss2: ", loss2)



    Current_accuracy = []
    # search the line including accuracy
    for line in file:
        m = re.search('Current_accuracy', line)
        if m:
            n = re.search('[0-9]+\.[0-9]+', line)           # 正则表达式
            if n is not None:
                Current_accuracy.append(float(n.group()))   # 提取精度数字
                print("Current_accuracy: ", Current_accuracy)


    file.close()

    x_values = list(range(578))
    y_ticks = list(np.linspace(0, 100, 20))         # 纵坐标的值，可以自己设置
    print('y_ticks: ', y_ticks)
    plt.plot(x_values, Current_accuracy, color='r', label='Current_accuracy')
    plt.yticks(y_ticks)
    plt.xlabel('batches')
    plt.grid()
    plt.title('bankcard')
    plt.show()

    # fig = plt.figure()
    # ax = fig.add_subplot(1, 1, 1)
    # x_values = list(range(578))
    # y_ticks = list(range(0, 2, 0.01))  # 纵坐标的值，可以自己设置。
    # ax.plot(x_values, loss1, color='r', label='Train_loss')
    # plt.yticks(y_ticks)  # 如果不想自己设置纵坐标，可以注释掉。
    # plt.grid()
    # ax.legend(loc='best')
    # ax.set_title('The loss curves')
    # ax.set_xlabel('batches')
    # fig.savefig('log/Train_loss.png')

if __name__ == '__main__':
    main()