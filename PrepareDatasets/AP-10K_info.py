"""
# --------------------------------------------------------
# @Project: 读取并预处理AP-10K数据集
# @Author : Hanle
# @E-mail : hanle@zju.edu.cn
# @Date   : 2023-07-24
# --------------------------------------------------------
"""

from pycocotools.coco import COCO
import numpy as np
import skimage.io as io
import matplotlib.pyplot as plt
import pylab
import os
from PIL import Image
from PIL import ImageDraw
import csv
pylab.rcParams['figure.figsize'] = (8.0, 10.0)

# initialize COCO api for keypoints annotations
dataDir = 'D:/Data/Pose/ap-10k/annotations'
dataType = 'ap10k-val-split3'
annFile = '{}/{}.json'.format(dataDir, dataType)
coco_kps = COCO(annFile)

csvPath = dataDir + '/TransferedLabels/AP_10K_info.csv'
csvFile = open(csvPath, 'w+', newline='')
keypointsWriter = csv.writer(csvFile)

# display COCO categories and super categories
cats = coco_kps.loadCats(coco_kps.getCatIds())
print(len(cats))
for i in range(len(cats)):
    print(i, cats[i]['name'], cats[i]['supercategory'])
    results = [str(i), cats[i]['name'], cats[i]['supercategory']]
    keypointsWriter.writerow(results)

# nms = [cat['name'] for cat in cats]
# nms2 = [cat['supercategory'] for cat in cats]
# keypointsWriter.writerow(nms)
# keypointsWriter.writerow(nms2)

