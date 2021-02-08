
import json
from collections import defaultdict

# delete the unused keypoints
DEL = [1, 2, 3, 4]  # left eye, right eye, left ear, right ear


def process_gt(in_file, out_file):
    with open(in_file, 'r') as f:
        data = json.load(f)  # dict

        data1 = data['annotations']
        json_result = defaultdict(list)

        for item in data1:
            res = {}
            res["image_name"] = item["image_name"]
            res["bbox"] = item["bbox"]
            xs = item['keypoints'][0::3]
            ys = item['keypoints'][1::3]
            label1 = item['keypoints'][2::3]

            keypoint = []
            label = []
            for i in range(len(xs)):
                if i not in DEL:
                    keypoint.append(xs[i])
            for i in range(len(ys)):
                if i not in DEL:
                    keypoint.append(ys[i])
            for i in range(len(label1)):
                if i not in DEL:
                    label.append(label1[i])
            res['keypoints'] = keypoint
            res['label'] = label
            json_result['images'].append(res)
        with open(out_file, 'w') as json_file:
            json_file.write(json.dumps(json_result))


if __name__ == "__main__":
    in_file = "../data/yoga_eval.json"
    out_file = "../data/gt/yoga_eval_gt.json"
    process_gt(in_file, out_file)

