
import numpy as np
import eval_helpers
from collections import OrderedDict


def computeDist(preFrames, gtFrames):
    # assert(len(gtFrames) == len(prFrames))

    nJoints = 13
    distAll = {}
    for pidx in range(nJoints):
        distAll[pidx] = np.zeros([0, 0])
    keypoints_gt,label = eval_helpers.eval1(gtFrames)
    keypoints_pre = eval_helpers.eval(preFrames)
    for imgidx in keypoints_gt.keys():
        gt = keypoints_gt[imgidx]
        pre = keypoints_pre[imgidx]
        head = gt[0]
        neck = ((gt[1][0] + gt[2][0]) / 2,
                (gt[1][1] + gt[2][1]) / 2)
        if imgidx in keypoints_pre.keys():
            for index in range(len(label)):
                if label[index] == 0:
                    distAll[index] = 0
                else:
                    pointGT = gt[index]
                    pointPre = pre[index]
                    d = np.linalg.norm(np.subtract(pointGT, pointPre))
                    headSize = eval_helpers.getHeadSize(head[0], neck[0], head[1], neck[1])
                    dnormal = d / headSize * 0.1
                    distAll[index] = np.append(distAll[index], [[dnormal]])
            return distAll


def computePCK(distAll, distThresh):
    pckAll = np.zeros([len(distAll) + 1, 1])
    nCorrect = 0
    nTotal = 13
    result = open("test_result.csv", "a")
    result.write(
        "folder_name,image_name,model,pckh_head,pckh_lShoulder,pckh_rShoulder,pckh_lElbow, pckh_rElbow,pckh_rWrist,pckh_rWrist,pckh_lHip, pckh_rHip,pckh_lKnee,pckh_rKnee,pckh_lAnkle,pckh_rAnkle,PCKH\n")
    result.close()
    for pidx in range(len(distAll)):
        idxs = np.argwhere(distAll[pidx] <= distThresh)
        pck = 100.0 * (1-distAll[pidx])
        pckAll[pidx, 0] = pck
        nCorrect += len(idxs)
    pckAll[-1, 0] = 100.0 * nCorrect / nTotal
    name_value = [('Shoulder', 0.5 * (pckAll[1] + pckAll[2])),
                  ('Elbow', 0.5 * (pckAll[3] + pckAll[4])),
                  ('Wrist', 0.5 * (pckAll[5] + pckAll[6])),
                  ('Hip', 0.5 * (pckAll[7] + pckAll[8])),
                  ('Knee', 0.5 * (pckAll[9] + pckAll[10])),
                  ('Ankle', 0.5 * (pckAll[11] + pckAll[12])),
                  ('PCKh', pckAll[-1])]
    name_value = OrderedDict(name_value)
    result = open("test_result.csv", "a+")
    result.write("{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{},{}\n".
                 format(preFramesAll, id, preFramesAll, pckAll[0], pckAll[1], pckAll[2], pckAll[3],
                        pckAll[4], pckAll[5], pckAll[6], pckAll[7], pckAll[8], pckAll[9], pckAll[10], pckAll[11],
                        pckAll[12], pckAll[13]))
    result.close()

    return name_value





def evaluatePCKh(gtFramesAll, prFramesAll):
    distThresh = 0.5

    # compute distances
    distAll = computeDist(gtFramesAll, prFramesAll)

    # compute PCK metric
    pckAll = computePCK(distAll, distThresh)

    return pckAll


if __name__ == '__main__':
    preFramesAll = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/poseval/data/tree_new.json"
    gtFramesAll = "/media/hkuit155/8221f964-4062-4f55-a2f3-78a6632f7418/Mobile-Pose/poseval/data/gt/tree_new_gt.json"
    pckAll = evaluatePCKh(preFramesAll, gtFramesAll)
    print(pckAll)