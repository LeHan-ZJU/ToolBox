"""
# --------------------------------------------------------
# @Project: 读取并预处理TigDog数据集
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2023-06-08
# --------------------------------------------------------
"""

import cv2
import numpy as np

# dir = 'D:\Data\Pose\SyntheticAnimals\horse_combineds5r5_texture'
#
# label = np.load(dir + '/1_SK_horse_skeleton_horse_idleB_anim_0.15_350.00_0.00_0.00.png_kpts.npy')
# img = cv2.imread(dir + '/1_SK_horse_skeleton_horse_idleB_anim_0.15_350.00_0.00_0.00.png_img.png')
# cv2.imshow('img', img)
# cv2.waitKey(0)
relation = [[0, 2], [2, 0], [0, 18], [18, 0], [1, 2], [2, 1],
            [1, 18], [18, 1], [3, 8], [8, 3], [4, 9], [9, 4],
            [5, 10], [10, 5], [6, 11], [11, 6], [7, 12], [12, 7],
            [7, 13], [13, 7], [7, 16], [16, 7], [7, 17], [17, 7],
            [8, 14], [14, 8], [9, 15], [15, 9], [10, 16], [16, 10],
            [11, 17], [17, 11], [12, 14], [14, 12], [12, 18], [18, 12],
            [13, 15], [15, 13], [13, 18], [18, 13]]
relation = np.array(relation)

w, h = relation.shape
temp = np.ones([40, 2]) * 19 + relation
temp = np.array(temp, dtype=np.int8)
print(relation)
print(w, h, temp)

