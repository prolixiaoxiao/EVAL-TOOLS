import torch
import numpy as np
from collections import OrderedDict

def cal_pckh(y_pred, y_true,refp=0.5):
    central=y_true[:,-11,:]+y_true[:,-12,:]
    head_size = norm(central - y_true[:,0,:], axis=1)
    # head_size = np.linalg.norm(head_size, axis=0)
    # head_size *= 0.6
    assert len(y_true) == len(head_size)
    num_samples = len(y_true)
    # for coco datasets, abandon eyes and ears keypoints
    used_joints = range(4,16)
    y_true = y_true[:, used_joints,:]
    y_pred = y_pred[:, used_joints,:]
    dist = np.zeros((num_samples,len(used_joints)))
    valid = np.zeros((num_samples,len(used_joints)))

    for i in range(num_samples):
        valid[i,:] = valid_joints(y_true[i])
        dist[i,:] = norm(y_true[i] - y_pred[i], axis=1) / head_size[i]

    pckh = dist * valid
    jnt_count = np.sum(valid,axis=1)
    jnt_count = np.ma.array(jnt_count, mask = False)
    jnt_ratio = jnt_count/np.sum(jnt_count).astype(np.float64)
    PCKh = np.ma.array(pckh, mask=False)
    # PCKh.mask[6:8] = True
    for i in range(num_samples):
        name_value = [('Shoulder', 0.5 * (PCKh[i][0] + PCKh[i][1])),
                      ('Elbow', 0.5 * (PCKh[i][2] + PCKh[i][3])),
                      ('Wrist', 0.5 * (PCKh[i][4] + PCKh[i][5])),
                      ('Hip', 0.5 * (PCKh[i][6] + PCKh[i][7])),
                      ('Knee', 0.5 * (PCKh[i][8] + PCKh[i][9])),
                      ('Ankle', 0.5 * (PCKh[i][10] + PCKh[i][11])),
                      ('PCKh', np.mean(PCKh[i][:])),
                      ('PCKh@0.5',np.sum(PCKh[i][:] > refp))]
        name_value = OrderedDict(name_value)
        print(name_value)
    return name_value


def norm(x, axis=None):
    return np.sqrt(np.sum(np.power(x, 2).numpy(), axis=axis))

def valid_joints(y, min_valid=0):
    def and_all(x):
        if x.all():
            return 1
        return 0

    return np.apply_along_axis(and_all, axis=1, arr=(y > min_valid))

if __name__ == '__main__':
    preds = torch.Tensor(([
        [[40., 44.],
         [40., 60.],
         [37., 54.],
         [38., 52.],
         [38., 65.],
         [31., 59.],
         [40., 63.],
         [38., 65.],
         [20., 57.],
         [30., 38.],
         [32., 55.],
         [26., 53.],
         [24., 41.],
         [36., 42.],
         [45., 65.],
         [36., 52.],
         [37., 51.]],

        [[35., 40.],
         [29., 30.],
         [34., 46.],
         [28., 34.],
         [37., 31.],
         [43., 35.],
         [49., 42.],
         [46., 27.],
         [13., 44.],
         [42., 34.],
         [40., 24.],
         [29., 43.],
         [34., 39.],
         [30., 41.],
         [33., 42.],
         [31., 47.],
         [16., 44.]]]))
    gt = torch.Tensor(([
        [[31., 47.],
         [48., 20.],
         [32., 46.],
         [32., 64.],
         [40., 48.],
         [42., 45.],
         [40., 58.],
         [43., 23.],
         [44., 46.],
         [41., 49.],
         [14., 20.],
         [46., 51.],
         [49., 50.],
         [57., 68.],
         [51., 23.],
         [48., 42.],
         [52., 22.]],

        [[33., 57.],
         [37., 25.],
         [22., 48.],
         [24., 27.],
         [46., 40.],
         [14., 55.],
         [30., 25.],
         [39., 63.],
         [54., 52.],
         [13., 54.],
         [30., 22.],
         [29., 38.],
         [16., 35.],
         [11., 48.],
         [23., 52.],
         [41., 53.],
         [33., 63.]]]))
    cal_pckh(preds, gt,refp=0.5)