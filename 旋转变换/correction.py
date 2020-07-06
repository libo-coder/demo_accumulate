# -*- coding: utf-8 -*-
"""
图片旋转变换
@author: libo
"""
import cv2
import numpy as np

def CalcEuclideanDistance(point1, point2):
    vec1 = np.array(point1)
    vec2 = np.array(point2)
    distance = np.linalg.norm(vec1 - vec2)      # linalg = linear（线性）+ algebra（代数），norm则表示范数
    return distance

####################################################################
# image_src: 原图
# resize_ratio: 缩放比，缺省为1.0不缩放
# diff_threshold： 平行四边形对应边相异程度容忍度，越大越粗放，最精确为1
# w,h：返回透视变换图像的宽高
# debug：可打开debug模式，看中间结果
####################################################################
def PerspectiveTransformationXH(image_src, resize_ratio=1.0, diff_threshold=1.2, w=430, h=270, debug=False):
    height = image_src.shape[1]
    width = image_src.shape[0]
    image_src = cv2.resize(image_src, (int(height*resize_ratio), int(width*resize_ratio)), interpolation=cv2.INTER_CUBIC)

    B_channel, G_channel, R_channel = cv2.split(image_src)  # 注意cv2.split()返回通道顺序

    # 蓝色通道阈值(调节好函数阈值为160时效果最好，太大一片白，太小干扰点太多)
    _, image = cv2.threshold(B_channel, 190, 255, cv2.THRESH_BINARY)

    # 寻找边缘
    edges = cv2.Canny(image, 50, 150, apertureSize=3)
    if debug:
        cv2.imshow("edges", edges)

    # 霍夫变换找直线两端点
    linesP = cv2.HoughLinesP(edges, 1, np.pi / 180, 60, minLineLength=60, maxLineGap=10)

    # 收集所有直线端点
    points_set = []
    for lineP in linesP:
        x1, y1, x2, y2 = lineP[0]
        points_set.append((x1, y1))
        points_set.append((x2, y2))
        if debug:
            cv2.line(image_src, (x1, y1), (x2, y2), (0, 0, 255), 2)
    # 分别按 axis=x, y 排序
    points_set_x_sorted = sorted(points_set, key=(lambda x: x[0]))
    points_set_y_sorted = sorted(points_set, key=(lambda x: x[1]))

    if debug:
        cv2.circle(image_src, points_set_x_sorted[0], 1, (255, 0, 0), 4)
        cv2.circle(image_src, points_set_x_sorted[-1], 1, (0, 255, 0), 4)
        cv2.circle(image_src, points_set_y_sorted[0], 1, (0, 0, 255), 4)
        cv2.circle(image_src, points_set_y_sorted[-1], 1, (255, 255, 0), 4)
        cv2.imshow("debug", image_src)

    if CalcEuclideanDistance(points_set_y_sorted[0], points_set_x_sorted[0]) < \
            diff_threshold * CalcEuclideanDistance(points_set_y_sorted[-1], points_set_x_sorted[-1]) and \
        diff_threshold * CalcEuclideanDistance(points_set_y_sorted[0],points_set_x_sorted[0]) > \
            CalcEuclideanDistance(points_set_y_sorted[-1], points_set_x_sorted[-1]) and \
        CalcEuclideanDistance(points_set_y_sorted[0],points_set_x_sorted[-1]) < \
            diff_threshold * CalcEuclideanDistance(points_set_y_sorted[-1], points_set_x_sorted[0]) and \
        diff_threshold * CalcEuclideanDistance(points_set_y_sorted[0],points_set_x_sorted[-1]) > \
            CalcEuclideanDistance(points_set_y_sorted[-1], points_set_x_sorted[0]):
        pass
    else:
        return image_src

    # 原图中卡片的四个角点
    if points_set_y_sorted[0][0] > points_set_y_sorted[-1][0]:
        pts1 = np.float32([points_set_x_sorted[0], points_set_y_sorted[0], points_set_y_sorted[-1], points_set_x_sorted[-1]])
    else:
        pts1 = np.float32([points_set_y_sorted[0], points_set_x_sorted[-1], points_set_x_sorted[0], points_set_y_sorted[-1]])
    # 变换后分别在左上、右上、左下、右下四个点
    pts2 = np.float32([[0, 0], [w, 0], [0, h], [w, h]])

    # 生成透视变换矩阵
    M = cv2.getPerspectiveTransform(pts1, pts2)     # 透视变换的四个点坐标的顺序不要错了！
    # 进行透视变换
    image_dst = cv2.warpPerspective(image_src, M, (w, h))
    return image_dst


if __name__ == '__main__':
    img0 = cv2.imread("image/test.jpg")
    img = PerspectiveTransformationXH(img0, resize_ratio=0.5)
    cv2.imwrite('image/cor_test.jpg', img)