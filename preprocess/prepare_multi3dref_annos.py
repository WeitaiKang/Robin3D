import json
import torch
import os
import sys
sys.path.append('.')
from tqdm import tqdm
import argparse
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from prompts.prompts import multi3dref_prompt, ID_format
import random
from collections import defaultdict
import string

parser = argparse.ArgumentParser()
parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ['train', 'val']:
    annos = json.load(open(f"/data/kangweitai/3D/annotation/multi3drefer_train_val/multi3drefer_{split}.json"))
    new_annos = []

    count_all = defaultdict(int)
    count_used = defaultdict(int)

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for i, anno in tqdm(enumerate(annos), total=len(annos)):
        scene_id = anno['scene_id']
        count_all[anno['eval_type']] += 1
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        gt_ids = anno['object_ids']
        caption = anno['description']
        if caption[-1] in string.punctuation:
            caption = caption[:-1]
        prompt = random.choice(multi3dref_prompt).replace("<description>", caption)
        pred_ids = []
        flag = 1
        for gt_id in gt_ids:
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
            if split == 'train' and (max_iou < args.train_iou_thres or max_id in pred_ids):
                flag = 0
                break
            pred_ids.append(max_id)
        if flag == 0:
            continue
        count_used[anno['eval_type']] += 1
        pred_ids = sorted(pred_ids)
        pred_id_strs = [ID_format.format(pred_id) for pred_id in pred_ids]
        if len(pred_ids) == 0:
            answer = "No."
        elif len(pred_ids) == 1:
            answer = f"Yes. {pred_id_strs[0]}."
        elif len(pred_ids) == 2:
            answer = f"Yes. {pred_id_strs[0]} and {pred_id_strs[1]}."
        else:
            answer = f"Yes. {', '.join(pred_id_strs[:-1])}, and {pred_id_strs[-1]}."
        if split == 'train':
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': 0,
                'prompt': prompt,
                'caption': answer,
                'eval_type': anno['eval_type']
            })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': 0,
                'prompt': prompt,
                'ref_captions': gt_ids,
                'eval_type': anno['eval_type']
            })

    print(f"Ori: {len(annos)}", count_all)
    print(f"new annos: {len(new_annos)}", count_used)

    print(f'--- new multi3dref_{segmentor}_{split}{version}.json has {len(new_annos)} annotations ---')
    save_path = os.path.join(args.save_dir, f"multi3dref_{segmentor}_{split}{version}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)
