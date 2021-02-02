import numpy as np
import eval_helpers
from collections import OrderedDict

def computeDist(gtFrames,prFrames):
    # assert(len(gtFrames) == len(prFrames))

    nJoints = 13
    distAll = {}
    for pidx in range(nJoints):
        distAll[pidx] = np.zeros([0,0])
    keypoints_gt = eval_helpers.eval(gtFrames)
    keypoints_pre = eval_helpers.eval(prFrames)
    for imgidx in keypoints_gt.keys():
        gt = keypoints_gt[imgidx]
        pre = keypoints_pre[imgidx]
        head = gt[0]
        neck = ((gt[1][0] + gt[2][0]) / 2,
                (gt[1][1] + gt[2][1]) / 2)
        if imgidx in keypoints_pre.keys():
            for index in range(len(pre)):
                pointGT = gt[index]
                pointPre = pre[index]
                d = np.linalg.norm(np.subtract(pointGT,pointPre))
                headSize = eval_helpers.getHeadSize(head[0], neck[0], head[1], neck[1])
                dnormal = d/headSize * 0.1
                distAll[index] = np.append(distAll[index],[[dnormal]])
            return distAll




def computePCK(distAll,distThresh):

    pckAll = np.zeros([len(distAll)+1,1])
    nCorrect = 0
    nTotal = 0
    for pidx in range(len(distAll)):
        idxs = np.argwhere(distAll[pidx] <= 0.85)
        pck = 100.0 * distAll[pidx]
        pckAll[pidx,0] = pck
        nCorrect += len(idxs)
        nTotal   += len(distAll[pidx])
    pckAll[-1,0] = 100.0*nCorrect/nTotal

    name_value = [('Shoulder', 0.5 * (pckAll[1] + pckAll[2])),
                  ('Elbow', 0.5 * (pckAll[3] + pckAll[4])),
                  ('Wrist', 0.5 * (pckAll[5] + pckAll[6])),
                  ('Hip', 0.5 * (pckAll[7] + pckAll[8])),
                  ('Knee', 0.5 * (pckAll[9] + pckAll[10])),
                  ('Ankle', 0.5 * (pckAll[11] + pckAll[12])),
                  ('PCKh@0.1', pckAll[-1])]
    name_value = OrderedDict(name_value)

    return name_value

    return pckAll


def evaluatePCKh(gtFramesAll,prFramesAll):

    distThresh = 0.5

    # compute distances
    distAll = computeDist(gtFramesAll,prFramesAll)

    # compute PCK metric
    pckAll = computePCK(distAll,distThresh)

    return pckAll
if __name__ == '__main__':
    preFramesAll = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Downloads/poseval-master/test/pre/tree.json"
    gtFramesAll = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Downloads/poseval-master/test/gt/tree_gt.json"
    pckAll = evaluatePCKh(preFramesAll,gtFramesAll)