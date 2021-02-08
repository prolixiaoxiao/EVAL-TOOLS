from collections import defaultdict
import numpy as np
import json

MIN_SCORE = -9999
MAX_TRACK_ID = 10000


class Joint:
    def __init__(self):
        self.count = 15
        self.right_ankle = 0
        self.right_knee = 1
        self.right_hip = 2
        self.left_hip = 3
        self.left_knee = 4
        self.left_ankle = 5
        self.right_wrist = 6
        self.right_elbow = 7
        self.right_shoulder = 8
        self.left_shoulder = 9
        self.left_elbow = 10
        self.left_wrist = 11
        self.neck = 12
        self.nose = 13
        self.head_top = 14


def getHeadSize(x1, y1, x2, y2):
    headSize = 0.8 * np.linalg.norm(np.subtract([x2, y2], [x1, y1]));
    return headSize


# compute recall/precision curve (RPC) values
def computeRPC(scores, labels, totalPos):
    precision = np.zeros(len(scores))
    recall = np.zeros(len(scores))
    npos = 0;

    idxsSort = np.array(scores).argsort()[::-1]
    labelsSort = labels[idxsSort];

    for sidx in range(len(idxsSort)):
        if (labelsSort[sidx] == 1):
            npos += 1
        # recall: how many true positives were found out of the total number of positives?
        recall[sidx] = 1.0 * npos / totalPos
        # precision: how many true positives were found out of the total number of samples?
        precision[sidx] = 1.0 * npos / (sidx + 1)

    return precision, recall, idxsSort
transform = list(zip(
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
        [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
    ))
def eval1(json_file):
    anno = json.load(open(json_file))
    keypoint = {}
    label = []
    for img_info in anno['images']:
        # images.append(img_info['image_name'])
        label = img_info['label']
        xs = img_info['keypoints'][0::2]
        ys = img_info['keypoints'][1::2]
        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (xs[idx], ys[idy])
            )
        keypoint[img_info['image_name']] = new_kp
        # keypoint[img_info['image_name']] = img_info['keypoints']

    return keypoint,label

def eval(json_file):
    anno = json.load(open(json_file))
    images = []
    keypoint = {}
    for img_info in anno['annotations']:
        images.append(img_info['image_name'])
        xs = img_info['keypoints'][0::3]
        ys = img_info['keypoints'][1::3]
        new_kp = []
        for idx, idy in transform:
            new_kp.append(
                (xs[idx], ys[idy])
            )

        keypoint[img_info['image_name']] = new_kp
    return keypoint
