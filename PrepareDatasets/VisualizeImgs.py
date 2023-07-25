"""
# --------------------------------------------------------
# @Project: 读取生成的csv格式标签，并按标签读取图像和标注，并可视化
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2023-07-24
# --------------------------------------------------------
"""
import re
import os
import csv
import cv2
import numpy as np


def read_labels(label_dir):
    # 读标签
    with open(label_dir, 'r') as f:
        reader = csv.reader(f)
        labels = list(reader)
        imgs_num = len(labels)
    print('imgs_num:', imgs_num)
    return labels, imgs_num


def read_keypoints(anns, num_points):
    points_all_gt = np.zeros([num_points + 2, 2])
    searchContext2 = '_'
    for n in range(num_points):
        numList = [m.start() for m in re.finditer(searchContext2, anns[n + 3])]
        point = [int(anns[n + 3][0:numList[0]]),
                 int(anns[n + 3][numList[0] + 1:numList[1]])]
        # resize
        point_resize = point
        point_resize[0] = point[0]
        point_resize[1] = point[1]
        points_all_gt[n, :] = point_resize
    # 读取box信息
    numList = [m.start() for m in re.finditer(searchContext2, anns[2])]
    box = [int(anns[2][0:numList[0]]),int(anns[2][numList[1] + 1:numList[2]]),
           int(anns[2][numList[0] + 1:numList[1]]), int(anns[2][numList[2] + 1:])]
    points_all_gt[num_points, :] = box[0:2]
    points_all_gt[num_points + 1, :] = box[2:4]
    return points_all_gt


def visualize(Img, points, num_points):
    r = np.arange(50, 255, int(205 / num_points))
    for i in range(num_points):
        cv2.circle(Img, (int(points[i][0]), int(points[i][1])), 2,
                   [int(r[num_points - i]), 20, int(r[i])], 2)

    return Img


if __name__ == "__main__":
    label_path = 'D:/Data/Pose/ap-10k/annotations/TransferedLabels/ap10k-trainval.csv'
    img_path = 'D:/Data/Pose/ap-10k/data/'
    # 读取测试数据列表
    Labels, imgs_num = read_labels(label_path)  # 读取标签
    for k in range(1, imgs_num):
        # 读取图像
        img_name = Labels[k][0]
        fn = os.path.join(img_path + img_name)
        img0 = cv2.imread(fn)
        cv2.imshow('img0', img0)
        keypoints = read_keypoints(Labels[k], 17)

        img = visualize(img0, keypoints[:17][:], 17)

        cv2.imshow('img', img)
        cv2.waitKey(0)
