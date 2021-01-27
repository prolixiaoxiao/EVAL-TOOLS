from collections import OrderedDict
import numpy as np
import json
import os
import math
from opt import opt

class poseevalpckh:
    def __init__(self):
        pass
    distance = []
    def eval(pred_coords):
        anno = json.load(open("poseval/data/gt/coach123_3_val_gt.json"))
        pre = json.load(open(pred_coords))
        images_gt = []
        keypoint_gt = {}
        images_pre = []
        keypoint_pre = {}
        transform = list(zip(
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12],
            [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
        ))
        for img_info in anno['images']:
            images_gt.append(img_info['image_name'])

            gt_xs = img_info['keypoints'][0::2]
            gt_ys = img_info['keypoints'][1::2]

            new_kp = []
            for idx, idy in transform:
                new_kp.append(
                    (gt_xs[idx], gt_ys[idy])
                )

            keypoint_gt[img_info['image_name']] = new_kp

        for img_info in pre['images']:
            images_pre.append(img_info['image_name'])

            prev_xs = img_info['keypoints'][0::2]
            prev_ys = img_info['keypoints'][1::2]

            new_kp1 = []
            for idx, idy in transform:
                new_kp1.append(
                    (prev_xs[idx], prev_ys[idy])
                )

            keypoint_pre[img_info['image_name']] = new_kp1


        for img_id in keypoint_gt.keys():
            groundtruth_anno = keypoint_gt[img_id]
            pre = keypoint_pre[img_id]
            head_gt = groundtruth_anno[0]

            neck_gt = (
                (groundtruth_anno[1][0] + groundtruth_anno[2][0]) / 2,
                (groundtruth_anno[1][1] + groundtruth_anno[2][1]) / 2)

            head_len = math.sqrt((head_gt[0] - neck_gt[0]) ** 2 + (head_gt[1] - neck_gt[1]) ** 2)
            d = []
            if img_id in keypoint_pre.keys():
                for index in range(len(pre)):
                    pred_x, pred_y = pre[index]
                    gt_x, gt_y = groundtruth_anno[index]
                    d.append(math.sqrt((pred_x - gt_x) ** 2 + (pred_y - gt_y) ** 2)/head_len*0.5)
                    pckh = d
                    PCKh = np.ma.array(pckh, mask=False)
                # name_value = [('Shoulder', 0.5 * (PCKh[index][1] + PCKh[index][2])),
                #               ('Elbow', 0.5 * (PCKh[index][4] + PCKh[index][3])),
                #               ('Wrist', 0.5 * (PCKh[index][6] + PCKh[index][5])),
                #               ('Hip', 0.5 * (PCKh[index][8] + PCKh[index][7])),
                #               ('Knee', 0.5 * (PCKh[index][10] + PCKh[index][9])),
                #               ('Ankle', 0.5 * (PCKh[index][12] + PCKh[index][11])),
                #               ('PCKh', np.mean(PCKh[index][:]))]
                name_value = [('Shoulder', 0.5 * (PCKh[1] + PCKh[2])),
                              ('Elbow', 0.5 * (PCKh[4] + PCKh[3])),
                              ('Wrist', 0.5 * (PCKh[6] + PCKh[5])),
                              ('Hip', 0.5 * (PCKh[8] + PCKh[7])),
                              ('Knee', 0.5 * (PCKh[10] + PCKh[9])),
                              ('Ankle', 0.5 * (PCKh[12] + PCKh[11])),
                              ('PCKh', np.mean(PCKh[:]))]
                name_value = OrderedDict(name_value)
                # print(name_value)
        pckh = np.mean(PCKh)
        print('PCKh',pckh)
                # return name_value
if __name__ == '__main__':
    eval = poseevalpckh
    resultpath = './poseval/results/'
    pred_coords = "poseval/data/coach123_3_val.json"
    PCKH = eval.eval(pred_coords)
    # txt_file = open(os.path.join(resultpath ), "w+")
    # txt_file.write("{}.pb: Average inference time :{}, PCKh:{}".
    #         format( str(PCKH)))
    # txt_file.close()