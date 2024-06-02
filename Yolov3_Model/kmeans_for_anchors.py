import glob
import random
import xml.etree.ElementTree as ET

import numpy as np


def cas_iou(box,cluster):
    x = np.minimum(cluster[:,0],box[0])
    y = np.minimum(cluster[:,1],box[1])

    intersection = x * y
    area1 = box[0] * box[1]

    area2 = cluster[:,0] * cluster[:,1]
    iou = intersection / (area1 + area2 -intersection)

    return iou

def avg_iou(box,cluster):
    return np.mean([np.max(cas_iou(box[i],cluster)) for i in range(box.shape[0])])


def kmeans(box,k):
    # How many boxes are there in total?
    row = box.shape[0]
    
    # The position of each point in each box
    distance = np.empty((row,k))
    
    # final clustering position
    last_clu = np.zeros((row,))

    np.random.seed()

    # Randomly select 5 point as cluster centers
    cluster = box[np.random.choice(row,k,replace = False)]
    # cluster = random.sample(row, k)
    while True:
        # Calculate the iou situation of each row being five points away.
        for i in range(row):
            distance[i] = 1 - cas_iou(box[i],cluster)
        
        # Take out the minimum point
        near = np.argmin(distance,axis=1)

        if (last_clu == near).all():
            break
        
        # Find the median point of each class
        for j in range(k):
            cluster[j] = np.median(
                box[near == j],axis=0)

        last_clu = near

    return cluster

def load_data(path):
    data = []
    # Find box for each xml
    for xml_file in glob.glob('{}/*xml'.format(path)):
        tree = ET.parse(xml_file)
        height = int(tree.findtext('./size/height'))
        width = int(tree.findtext('./size/width'))
        # For each target get its width and height
        for obj in tree.iter('object'):
            xmin = int(float(obj.findtext('bndbox/xmin'))) / width
            ymin = int(float(obj.findtext('bndbox/ymin'))) / height
            xmax = int(float(obj.findtext('bndbox/xmax'))) / width
            ymax = int(float(obj.findtext('bndbox/ymax'))) / height

            xmin = np.float64(xmin)
            ymin = np.float64(ymin)
            xmax = np.float64(xmax)
            ymax = np.float64(ymax)
            # Get width and height
            data.append([xmax-xmin,ymax-ymin])
    return np.array(data)


if __name__ == '__main__':
    # Running this program will calculate the xml of './VOCdevkit/VOC2007/Annotations'
    # yolo_anchors.txt will be generated
    SIZE = 416
    anchors_num = 9
    # To load the data set, you can use the VOC xml
    path = r'./VOCdevkit/VOC2007/Annotations'
    
    # Load all xml
    # The storage format is width and height converted into proportions.
    data = load_data(path)
    
    # Use k-clustering algorithm
    out = kmeans(data,anchors_num)
    out = out[np.argsort(out[:,0])]
    print('acc:{:.2f}%'.format(avg_iou(data,out) * 100))
    print(out*SIZE)
    data = out*SIZE
