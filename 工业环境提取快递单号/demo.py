# coding=utf-8
import cv2
import numpy as np
from PIL import Image
import os
import tqdm

def distance(x0, y0, x1, y1):
    dis = int(pow(pow(x1-x0, 2) + pow(y1-y0, 2), 0.5))
    return dis

def solve(box):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    cx = (x1 + x3 + x2 + x4) / 4.0
    cy = (y1 + y3 + y4 + y2) / 4.0
    w = (np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2) + np.sqrt((x3 - x4) ** 2 + (y3 - y4) ** 2)) / 2
    h = (np.sqrt((x2 - x3) ** 2 + (y2 - y3) ** 2) + np.sqrt((x1 - x4) ** 2 + (y1 - y4) ** 2)) / 2
    sinA = (h * (x1 - cx) - w * (y1 - cy)) * 1.0 / (h * h + w * w) * 2
    # print('sinA=', sinA)
    if sinA < -1:
        sinA = -1
    if sinA > 1:
        sinA = 1
    angle = np.arcsin(sinA)
    return angle, w, h, cx, cy

def rotate_cut_img(im, degree, box, w, h, leftAdjust=False, rightAdjust=False, alph=0.2):
    x1, y1, x2, y2, x3, y3, x4, y4 = box[:8]
    x_center, y_center = np.mean([x1, x2, x3, x4]), np.mean([y1, y2, y3, y4])
    degree_ = degree * 180.0 / np.pi
    right = 0
    left = 0
    if rightAdjust:
        right = 1
    if leftAdjust:
        left = 1

    box = (max(1, x_center - w / 2 - left * alph * (w / 2)), y_center - h / 2,
           min(x_center + w / 2 + right * alph * (w / 2), im.size[0] - 1), y_center + h / 2)

    newW = box[2] - box[0]
    newH = box[3] - box[1]
    tmpImg = im.rotate(degree_, center=(x_center, y_center)).crop(box)
    return tmpImg, newW, newH

img_fns = os.listdir('./tp/')
img_fns.sort()

if __name__ == '__main__':
    for i, img_fn in enumerate(img_fns):
        _, fn = os.path.split(img_fn)
        bfn, ext = os.path.splitext(fn)
        if ext.lower() not in ['.jpg', '.png', '.bmp']:
            continue
        print('*' * 80)
        print(bfn)
        src_img = cv2.imread('./tp/' + fn)
        # src_img = cv2.imread('./tp/' + '6A0248APAK00020_20200716_181005_601.jpg')
        img_h, img_w = src_img.shape[:2]

        img_gray = cv2.cvtColor(src_img, cv2.COLOR_BGR2GRAY)
        img_median = cv2.medianBlur(img_gray, 5)
        img_canny = cv2.Canny(img_median, 50, 150)
        # cv2.imwrite('./debug/img_canny.jpg', img_canny)
        kernel = np.ones((9, 9), np.uint8)
        dilation = cv2.dilate(img_canny, kernel, iterations=1)
        # cv2.imwrite('./debug/dilation.jpg', dilation)
        contours, hierarchy = cv2.findContours(dilation, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)  # cv2.RETR_EXTERNAL, RETR_TREE

        img_copy = src_img.copy()
        img_copy2 = src_img.copy()
        block_contours = []
        for contour in contours:
            target = True
            area = cv2.contourArea(contour)

            if area < float(img_h * img_w) / 50.0 or area > img_h * img_w * 0.90:
                target = False

            if not target:
                continue

            rect = cv2.minAreaRect(contour)
            print(rect)
            box = np.int0(cv2.boxPoints(rect))      # # 获得矩形角点
            # cv2.drawContours(img_copy2, [box], -1, (0, 0, 255), 2)
            # cv2.imwrite('./debug/contour_rect_' + str(i) + '.jpg', img_copy2)

            box = box.reshape((8,)).tolist()

            for bbox in box:
                if bbox < 0:
                    target = False
            if not target:
                continue

            rect_h = int(rect[1][0])
            rect_w = int(rect[1][1])
            aspect_ratio = float(max(rect_h, rect_w)) / min(rect_h, rect_w)        # 最大宽高比
            if aspect_ratio > 4:
                target = False
            rect_area = rect_h * rect_w
            if area < rect_area * 0.90:
                target = False
            if not target:
                continue

            ################# add：2020.07.19 #################
            # # 极点寻找
            # leftmost = tuple(contour[contour[:, :, 0].argmin()][0])
            # cv2.circle(img_copy2, leftmost, 5, [0, 0, 255], -1)
            # rightmost = tuple(contour[contour[:, :, 0].argmax()][0])
            # cv2.circle(img_copy2, rightmost, 5, [0, 0, 255], -1)
            # topmost = tuple(contour[contour[:, :, 1].argmin()][0])
            # cv2.circle(img_copy2, topmost, 5, [0, 0, 255], -1)
            # bottommost = tuple(contour[contour[:, :, 1].argmax()][0])
            # cv2.circle(img_copy2, bottommost, 5, [0, 0, 255], -1)
            #
            # text1 = 'Leftmost: ' + str(leftmost) + ' Rightmost: ' + str(rightmost)
            # text2 = 'Topmost: ' + str(topmost) + ' Bottommost: ' + str(bottommost)
            #
            # font = cv2.FONT_HERSHEY_SIMPLEX  # 设置字体样式
            # cv2.putText(img_copy2, text1, (10, 30), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.putText(img_copy2, text2, (10, 60), font, 0.5, (0, 255, 0), 1, cv2.LINE_AA, 0)
            # cv2.imwrite('./debug/contour_test.jpg', img_copy2)
            ###################################################

            ################# add：2020.07.19 #################
            # 凸包检测
            # points = cv2.convexHull(contour)
            # total = len(points)
            # for i in range(total):
            #     x1, y1 = points[i % total][0]
            #     x2, y2 = points[(i + 1) % total][0]
            #     cv2.circle(img_copy2, (x1, y1), 4, (255, 0, 0), 2, 8, 0)
            #     cv2.line(img_copy2, (x1, y1), (x2, y2), (0, 0, 255), 2, 8, 0)
            # cv2.imwrite('./debug/contour_test.jpg', img_copy2)
            ###################################################

            ################# add：2020.07.19 #################
            # 多边形拟合
            epsilon = 0.1 * cv2.arcLength(contour, True)
            approx = cv2.approxPolyDP(contour, epsilon, True)
            # print(approx)

            cv2.polylines(img_copy2, [approx], True, (0, 0, 255), 2)
            cv2.imwrite('./debug/contour_approx_' + str(i) + '.jpg', img_copy2)

            if len(approx) != 4:
                continue

            approx_list = approx.reshape((8,))
            # print(approx_list)
            (x1, y1) = (approx_list[0], approx_list[1])
            (x2, y2) = (approx_list[2], approx_list[3])
            (x3, y3) = (approx_list[4], approx_list[5])
            (x4, y4) = (approx_list[6], approx_list[7])

            box_tmp = [(x1, y1), (x2, y2), (x3, y3), (x4, y4)]

            index = np.argsort(box_tmp, axis=0)
            left = [box_tmp[index[0][0]], box_tmp[index[1][0]]]
            right = [box_tmp[index[2][0]], box_tmp[index[3][0]]]
            # print(left, right)

            if left[0][1] < left[1][1]:
                l_t = left[0]
                l_b = left[1]
            else:
                l_t = left[1]
                l_b = left[0]

            if right[0][1] < right[1][1]:
                r_t = right[0]
                r_b = right[1]
            else:
                r_t = right[1]
                r_b = right[0]

            # index2 = np.argsort(left, axis=-1)
            # l_t = left[index2[0][0]]
            # l_b = left[index2[0][1]]
            # print('l_t, l_b: ', l_t, l_b)

            # index3 = np.argsort(right, axis=-1)
            # r_t = right[index3[0][0]]
            # r_b = right[index3[0][1]]
            # print('r_t, r_b: ', r_t, r_b)

            ###################################################

            box = [l_t[0], l_t[1], r_t[0], r_t[1], r_b[0], r_b[1], l_b[0], l_b[1]]
            # box = [box[2], box[3], box[4], box[5], box[6], box[7], box[0], box[1]]


            ### 仿射变换
            # degree, roi_w, roi_h, _, _ = solve(box)
            # partImg, newW, newH = rotate_cut_img(Image.fromarray(np.array(src_img)), degree, box, roi_w, roi_h)

            ### 透视变换
            result_w = distance(box[0], box[1], box[2], box[3])
            result_h = distance(box[0], box[1], box[6], box[7])
            p1 = np.float32([(box[0], box[1]), (box[2], box[3]), (box[6], box[7]), (box[4], box[5])])
            p2 = np.float32([(0, 0), (result_w, 0), (0, result_h), (result_w, result_h)])
            M = cv2.getPerspectiveTransform(p1, p2)  # 变换矩阵
            result = cv2.warpPerspective(src_img, M, (result_w, result_h))

            if result_h >= result_w:
                cv2.imwrite('./debug/result_' + str(i) + '.jpg', result)
            else:
                result = np.rot90(result)
                cv2.imwrite('./debug/result_' + str(i) + '.jpg', result)

            # if newH >= newW:
            #     partImg.save('./debug/rotate_' + str(i) + '.jpg')
            # else:
            #     partImg = partImg.rotate(90, expand=1)
            #     partImg.save('./debug/rotate_' + str(i) + '.jpg')

            block_contours.append(contour)

        cv2.drawContours(img_copy, block_contours, -1, (0, 255, 0), 2)
        cv2.imwrite('./debug/contour_' + str(i) + '.jpg', img_copy)
        # raise