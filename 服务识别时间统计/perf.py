#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import argparse
import logging
import sys
import multiprocessing
import time
import os
import random
import base64
import json
import requests
import glob

test_list = glob.glob('/multi_test/test_0702_/hgx/*')   # 测试数据集路径


def test_func(count, host, port, i):
    pid = os.getpid()

    logger = logging.getLogger('root.pid%d' % i)
    logger.setLevel(logging.INFO)
    hdlr = logging.FileHandler('./%d.log' % pid)
    logger.addHandler(hdlr)

    max_http_time = 0.0
    min_http_time = sys.float_info.max
    avg_http_time = 0.0

    for i in range(count):
        img_path = random.choice(test_list)
        with open(img_path, 'rb') as img:
            mat = img.read()
        start_time = time.time()
        data = base64.b64encode(mat).decode()
        res = requests.post('http://'+host+':'+port+'/ocr', data=json.dumps({'image': data, 'type': '0'}, ensure_ascii=False))
        # res = requests.post('http://' + host + ':' + port + '/icr/recognize_multi_table_raw', data=mat)
        # print(res)
        http_time = time.time() - start_time
        print('cost time:', http_time)

        # logger.info('ocr path: %s, http time: %.3f, result text: %d' % (img_path, http_time, json.loads(res.text)['words_result_num']))
        print('[PID' + str(pid) + ']', img_path, res)
        # print('res:',res.text)
        if http_time > max_http_time:
            max_http_time = http_time
        if http_time < min_http_time:
            min_http_time = http_time
        avg_http_time += http_time

    # logger.info('max time: %.3f, min time: %.3f, avg time: %.3f' % (max_http_time, min_http_time, avg_http_time/count))
    print('max time: %.3f, min time: %.3f, avg time: %.3f' % (max_http_time, min_http_time, avg_http_time / count))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='ocr perf test')
    parser.add_argument('procs', help='testing processes', type=int)        # 进程数
    parser.add_argument('count', help='testing loopcount', type=int)        # 测试数据集中测试次数
    parser.add_argument('host', help='host of server')
    parser.add_argument('port', help='port of server')
    args = parser.parse_args()

    main_log = logging.getLogger('main')
    main_log.setLevel(logging.INFO)
    main_hdlr = logging.StreamHandler(sys.stdout)
    main_log.addHandler(main_hdlr)

    p = multiprocessing.Pool(args.procs)
    for i in range(args.procs):
        p.apply_async(test_func, args=(args.count, args.host, args.port, i))
        main_log.info('[%s]test%d running' % (time.asctime(time.localtime(time.time())), i))
        time.sleep(1)  #
    p.close()

    try:
        p.join()
    finally:
        main_log.info('[%s]testing over' % (time.ctime(time.mktime(time.localtime()))))
