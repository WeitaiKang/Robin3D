import numpy as np
import json
import sys
sys.path.append('.')
import torch
import random
from tqdm import tqdm
from collections import defaultdict
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
import os


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

unwanted_words = ["wall", "ceiling", "floor", "object", "item"]

segmentor = args.segmentor
version = args.version

def random_change_object2rest(input_object, object_corpus):
    if random.random() < 0.5:
        return True, random.choice(list(object_corpus - set([input_object])))
    else:
        return False, input_object

def get_qa(match_gt_predids, scannet_class_labels, object_corpus):
    gt_num = len(match_gt_predids)
    valid_gt_ids = [i for i in range(gt_num) if match_gt_predids[i] != -1]
    if len(valid_gt_ids) == 0:
        return "", "", -1
    # random choose one gt_id w/ match_gt_predids[gt_id] != -1
    gt_id = random.choice(valid_gt_ids)
    # get object class
    class_label = scannet_class_labels[gt_id]
    # get matched pred_id
    pred_id = int(match_gt_predids[gt_id])
    # random change results
    if_change, change_class_label = random_change_object2rest(class_label, object_corpus)
    q = f"<OBJ{pred_id:03}> is a {change_class_label}"
    if if_change:
        a = f"No, <OBJ{pred_id:03}> is a {class_label}"
        gt_id = -1
    else:
        a = f"Yes"
    return q, a, gt_id

for split in ["train"]:
    new_annos = []

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for scene_id in tqdm(scannet_attrs.keys(), total=len(scannet_attrs)):
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]['locs']
        scannet_locs = scannet_attrs[scene_id]['locs']
        instance_num = len(instance_locs)
        gt_num = len(scannet_locs)
        scannet_class_labels = scannet_attrs[scene_id]['objects']

        flag = 1
        match_gt_predids = np.ones(gt_num) * -1
        match_pred_ids = []
        object_corpus = set()
        collected_prompts = []
        for gt_id in range(gt_num):
            class_label = scannet_class_labels[gt_id]
            if any(x in class_label for x in unwanted_words):
                continue
            object_corpus.add(class_label)

            max_iou, max_id = -1, -1
            for pred_id in range(instance_num):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[gt_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id
            if split == 'train':
                if max_iou >= args.train_iou_thres:
                    if max_id in match_pred_ids:
                        flag = 0
                        break
                    else:
                        match_pred_ids.append(max_id)
                        match_gt_predids[gt_id] = max_id
        if flag == 0:
            continue
        
        # for _ in range(20):
        for _ in range(20*5):
            current_match_gt_predids = match_gt_predids.copy()
            # random choose the number of object to question
            # x = random.choice([1, 2, 3])
            x = random.choice([1]*5 + [2]*4 + [3]*3 + [4]*2 + [5]*1)
            q_list, a_list = [], []
            for _ in range(x):
                q, a, used_gtid = get_qa(current_match_gt_predids, scannet_class_labels, object_corpus)
                if q != "":
                    q_list.append(q)
                    a_list.append(a)
                if used_gtid != -1:
                    current_match_gt_predids[used_gtid] = -1
            
            if len(q_list) > 0:
                prompt = f"Is it correct that {'; '.join(q_list)}?"
                caption = f"{'; '.join(a_list)}."
                if "_".join(sorted(q_list)) not in collected_prompts:
                    collected_prompts.append("_".join(sorted(q_list)))
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': 0,
                        'prompt': prompt,
                        'caption': caption
                    })
        
    save_path = os.path.join(args.save_dir, f"partial_obj_align_{segmentor}_{split}{version}_scale.json")
    print(f'--- new partial_obj_align_{segmentor}_{split}{version}.json has {len(new_annos)} annotations ---')
    with open(save_path, 'w') as f:
        json.dump(new_annos, f, indent=4)

#iou50
# --- new partial_obj_align_mask3d_train.json has 22741 annotations ---