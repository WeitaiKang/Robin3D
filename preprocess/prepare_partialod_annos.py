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

def random_get_object(object_corpus, global_object_corpus):
    if random.random() < 0.5:
        return True, random.choice(list(global_object_corpus - object_corpus))
    else:
        return False, ""

def get_qa(match_gt_predids, scannet_class_labels, 
           class_label, object_corpus, global_object_corpus):
    gt_num = len(match_gt_predids)
    if_change, change_class_label = random_get_object(object_corpus, global_object_corpus)
    if if_change:
        q = f"{change_class_label}"
        a = "No"
    else:
        q = f"{class_label}"
        pred_ids = [int(match_gt_predids[i]) for i in range(gt_num) 
                    if scannet_class_labels[i] == class_label]
        assert -1 not in pred_ids, f"{class_label}"
        pred_ids = sorted(pred_ids)
        pred_ids = [f'<OBJ{pred_id:03}>' for pred_id in pred_ids]
        if len(pred_ids) == 1:
            a = f"Yes. {pred_ids[0]}"
        elif len(pred_ids) == 2:
            a = f"Yes. {pred_ids[0]} and {pred_ids[1]}"
        else:
            a = f"Yes. {', '.join(pred_ids[:-1])}, and {pred_ids[-1]}"
    return q, a, if_change


for split in ["train"]:
    new_annos = []

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    global_object_corpus = set()
    for scene_id in scannet_attrs.keys():
        # global_object_corpus.update(set(scannet_attrs[scene_id]['objects']))
        for obj in scannet_attrs[scene_id]['objects']:
            if any(x in obj for x in unwanted_words):
                continue
            global_object_corpus.add(obj)

    for scene_id in tqdm(scannet_attrs.keys(), total=len(scannet_attrs)):
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]['locs']
        scannet_locs = scannet_attrs[scene_id]['locs']
        instance_num = len(instance_locs)
        gt_num = len(scannet_locs)
        scannet_class_labels = scannet_attrs[scene_id]['objects']

        flag = 1
        match_gt_predids = np.ones(gt_num, dtype=int) * -1
        match_pred_ids = []
        object_corpus = []
        collected_prompts = []
        for gt_id in range(gt_num):
            class_label = scannet_class_labels[gt_id]
            if any(x in class_label for x in unwanted_words):
                continue
            object_corpus.append(class_label)

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
                        match_gt_predids[gt_id] = int(max_id)
        if flag == 0:
            continue

        valid_object = [scannet_class_labels[i] for i in range(gt_num) if match_gt_predids[i] != -1]
        valid_object_le3 = set()
        for obj in valid_object:
            # if valid_object.count(obj) <= 3 and valid_object.count(obj) == object_corpus.count(obj):
            if valid_object.count(obj) <= 10 and valid_object.count(obj) == object_corpus.count(obj):
                valid_object_le3.add(obj)

        object_corpus = set(object_corpus)
        valid_object_le3 = list(valid_object_le3)
        if len(valid_object_le3) == 0:
            continue
        
        # for _ in range(20):
        for _ in range(100):
            current_valid_object_le3 = valid_object_le3.copy()
            # random choose the number of object to question
            # x = random.choice([1,1,1,1,2,2,3])
            x = random.choice([1]*5 + [2]*4 + [3]*3 + [4]*2 + [5]*1)
            q_list, a_list = [], []
            for _ in range(x):
                select_class = random.choice(current_valid_object_le3)
                q, a, if_change = get_qa(match_gt_predids, scannet_class_labels, 
                                         select_class, object_corpus, global_object_corpus)
                if not if_change:
                    current_valid_object_le3 = [obj for obj in current_valid_object_le3 if obj != select_class]
                q_list.append(q)
                a_list.append(a)
                if len(current_valid_object_le3) == 0:
                    break
            
            if len(q_list) > 0:
                prompt = f"Can you find any {'; '.join(q_list)}?"
                caption = f"{'; '.join(a_list)}."
                if "_".join(sorted(q_list)) not in collected_prompts:
                    collected_prompts.append("_".join(sorted(q_list)))
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': 0,
                        'prompt': prompt,
                        'caption': caption
                    })
        
    save_path = os.path.join(args.save_dir, f"partial_od_{segmentor}_{split}{version}_scale.json")
    print(f'--- new partial_od_{segmentor}_{split}{version}.json has {len(new_annos)} annotations ---')
    with open(save_path, 'w') as f:
        json.dump(new_annos, f, indent=4)

#iou50
# --- new partial_od_mask3d_train.json has 21143 annotations ---