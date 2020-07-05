# -*- coding: utf-8 -*-
"""
OCR 图像有关处理函数
@author: libo
"""
import numpy as np
import cv2

####################################### 图像框、线绘制 start #######################################
def draw_boxes(im, bboxes, color=(0, 0, 0)):
    tmp = np.copy(im)
    h, w = im.shape[:2]
    thick = int((h + w) // 300)
    for i, box in enumerate(bboxes):
        x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
        cx = np.mean([x1, x2, x3, x4])
        cy = np.mean([y1, y2, y3, y4])
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), color, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x2), int(y2)), (int(x3), int(y3)), color, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x3), int(y3)), (int(x4), int(y4)), color, 1, lineType=cv2.LINE_AA)
        cv2.line(tmp, (int(x4), int(y4)), (int(x1), int(y1)), color, 1, lineType=cv2.LINE_AA)
        mess = str(i)
        cv2.putText(tmp, mess, (int(cx), int(cy)), 0, 1e-3 * h, color, thick // 2)
    return tmp


def draw_lines(im, bboxes, color=(0, 0, 0), lineW=3):
    tmp = np.copy(im)
    h, w = im.shape[:2]
    i = 0
    for box in bboxes:
        x1, y1, x2, y2 = box
        cv2.line(tmp, (int(x1), int(y1)), (int(x2), int(y2)), color, lineW, lineType=cv2.LINE_AA)
        i += 1
    return tmp
####################################### 图像框、线绘制 end #######################################



####################################### 直线处理 start ##########################################
def fit_line(p1, p2):
    """ 直线一般方程
    A = Y2 - Y1
    B = X1 - X2
    C = X2 * Y1 - X1 * Y2
    AX + BY + C=0
    """
    x1, y1 = p1
    x2, y2 = p2
    A = y2 - y1
    B = x1 - x2
    C = x2 * y1 - x1 * y2
    return A, B, C


def line_point_line(point1, point2):
    """ 求解两条直线的交点
    A1x + B1y + C1=0
    A2x + B2y + C2=0
    x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
    y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
    """
    A1, B1, C1 = fit_line(point1[0], point1[1])
    A2, B2, C2 = fit_line(point2[0], point2[1])
    x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
    y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
    return x, y


def sqrt(p1, p2):
    return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)


def point_to_points(p, points, alpha=10):
    """ 点到点之间的距离 """
    sqList = [sqrt(p, point) for point in points]
    if max(sqList) < alpha:
        return True
    else:
        return False


def point_line_cor(p, A, B, C):
    """ 判断点与之间的位置关系  一般式直线方程(Ax+By+c)=0 """
    x, y = p
    r = A * x + B * y + C
    return r


def line_to_line(points1, points2, alpha=10):
    """ 线段之间的距离 """
    x1, y1, x2, y2 = points1
    ox1, oy1, ox2, oy2 = points2
    A1, B1, C1 = fit_line((x1, y1), (x2, y2))
    A2, B2, C2 = fit_line((ox1, oy1), (ox2, oy2))
    flag1 = point_line_cor([x1, y1], A2, B2, C2)
    flag2 = point_line_cor([x2, y2], A2, B2, C2)
    if (flag1 > 0 and flag2 > 0) or (flag1 < 0 and flag2 < 0):
        x = (B1 * C2 - B2 * C1) / (A1 * B2 - A2 * B1)
        y = (A2 * C1 - A1 * C2) / (A1 * B2 - A2 * B1)
        p = (x, y)
        r0 = sqrt(p, (x1, y1))
        r1 = sqrt(p, (x2, y2))
        if min(r0, r1) < alpha:
            if r0 < r1:
                points1 = [p[0], p[1], x2, y2]
            else:
                points1 = [x1, y1, p[0], p[1]]
    return points1

####################################### 直线处理 end ############################################


################################### 图像根据角度旋转 start #######################################
def rotate_bound(images, angle):
    # grab the dimensions of the image and then determine the center
    (h, w) = images[0].shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D((cX, cY), -angle, 1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    return [cv2.warpAffine(image, M, (nW, nH)) for image in images]

#################################### 图像根据角度旋转 end #######################################
