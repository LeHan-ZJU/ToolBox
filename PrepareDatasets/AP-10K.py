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

# display COCO categories and super categories
cats = coco_kps.loadCats(coco_kps.getCatIds())
nms = [cat['name'] for cat in cats]
keypoints_cats = cats[0]['keypoints']
print('AO-10K keypoints categories:', keypoints_cats)

print('AO-10K categories: \n{}\n'.format(' '.join(nms)))

nms2 = set([cat['supercategory'] for cat in cats])
print('AP-10K supercategories: \n{}'.format(' '.join(nms)))

# get all images containing given categories, select one at random
catIds = coco_kps.getCatIds(catNms=['person'])
imgIds = coco_kps.getImgIds(catIds=catIds)
print('there are %d images containing human' % len(imgIds))


def getBndboxKeypointsGT():
    # csvPath = dataDir + '/TransferedLabels/' + dataType + '.csv'
    # csvFile = open(csvPath, 'w+', newline='')
    csvPath = dataDir + '/TransferedLabels/ap10k-trainval.csv'
    csvFile = open(csvPath, 'a+', newline='')
    keypointsWriter = csv.writer(csvFile)
    # firstRow = ['imageName', 'categories', 'supercategory', 'bndbox']
    # # print(firstRow)
    # for i in range(len(keypoints_cats)):
    #     firstRow.append(keypoints_cats[i])
    # keypointsWriter.writerow(firstRow)

    for i in range(len(imgIds)):
        imageNameTemp = coco_kps.loadImgs(imgIds[i])[0]
        imageName = imageNameTemp['file_name']#.encode('raw_unicode_escape')
        img = coco_kps.loadImgs(imgIds[i])[0]
        annIds = coco_kps.getAnnIds(imgIds=img['id'], catIds=catIds, iscrowd=None)
        anns = coco_kps.loadAnns(annIds)
        # print(anns)
        personNumber = len(anns)
        # "keypoints": 每个关键点标签由三个数字组成，前两个是xy坐标，第三个为v，解释如下：
        # 如果关键点在物体segment内，则认为可见。
        # v=0 表示这个关键点没有标注（这种情况下x=y=v=0）；v=1 表示这个关键点标注了但是不可见(被遮挡了）；v=2 表示这个关键点标注了同时也可见
        # for j in range(personNumber):
        if personNumber == 1:
            category_id = anns[0]['category_id']
            bndbox = anns[0]['bbox']
            keyPoints = anns[0]['keypoints']
            keypointsRow = [imageName, str(category_id),
                            str(bndbox[0])+'_'+str(bndbox[1])+'_'+str(bndbox[2])+'_'+str(bndbox[3]),
                            str(keyPoints[0])+'_'+str(keyPoints[1])+'_'+str(keyPoints[2]),
                            str(keyPoints[3])+'_'+str(keyPoints[4])+'_'+str(keyPoints[5]),
                            str(keyPoints[6])+'_'+str(keyPoints[7])+'_'+str(keyPoints[8]),
                            str(keyPoints[9])+'_'+str(keyPoints[10])+'_'+str(keyPoints[11]),
                            str(keyPoints[12])+'_'+str(keyPoints[13])+'_'+str(keyPoints[14]),
                            str(keyPoints[15])+'_'+str(keyPoints[16])+'_'+str(keyPoints[17]),
                            str(keyPoints[18])+'_'+str(keyPoints[19])+'_'+str(keyPoints[20]),
                            str(keyPoints[21])+'_'+str(keyPoints[22])+'_'+str(keyPoints[23]),
                            str(keyPoints[24])+'_'+str(keyPoints[25])+'_'+str(keyPoints[26]),
                            str(keyPoints[27])+'_'+str(keyPoints[28])+'_'+str(keyPoints[29]),
                            str(keyPoints[30])+'_'+str(keyPoints[31])+'_'+str(keyPoints[32]),
                            str(keyPoints[33])+'_'+str(keyPoints[34])+'_'+str(keyPoints[35]),
                            str(keyPoints[36])+'_'+str(keyPoints[37])+'_'+str(keyPoints[38]),
                            str(keyPoints[39])+'_'+str(keyPoints[40])+'_'+str(keyPoints[41]),
                            str(keyPoints[42])+'_'+str(keyPoints[43])+'_'+str(keyPoints[44]),
                            str(keyPoints[45])+'_'+str(keyPoints[46])+'_'+str(keyPoints[47]),
                            str(keyPoints[48])+'_'+str(keyPoints[49])+'_'+str(keyPoints[50])]
            keypointsWriter.writerow(keypointsRow)

    csvFile.close()

if __name__ == "__main__":
    print('Writing bndbox and keypoints to csv files..."')
    getBndboxKeypointsGT()
