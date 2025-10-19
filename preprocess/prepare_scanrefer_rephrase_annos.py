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
from prompts.prompts import grounding_prompt, scanrefer_prompt
import string
import os


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

for split in ["train"]:
    annos = json.load(open(f"/data/kangweitai/3D/annotation/gpt_rephrase/scanrefer_train_rephrase_merged.json", "r"))
    annos = sorted(annos, key=lambda p: f"{p['scene_id']}_{int(p['object_id']):03}")
    new_annos = []

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    iou25_count = 0
    iou50_count = 0
    for i, anno in tqdm(enumerate(annos), total=len(annos)):
        scene_id = anno['scene_id']
        obj_id = int(anno['object_id'])
        desc = anno['description']
        if desc[-1] in string.punctuation:
            desc = desc[:-1]
        prompt = random.choice(scanrefer_prompt).replace('<description>', desc)
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        max_iou, max_id = -1, -1
        for pred_id in range(instance_num):
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)
            if iou > max_iou:
                max_iou = iou
                max_id = pred_id

        if max_iou >= 0.25:
            iou25_count += 1
        if max_iou >= 0.5:
            iou50_count += 1
        if split == "train":
            if max_iou >= args.train_iou_thres:
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": max_id,
                    "caption": f"<OBJ{max_id:03}>.",
                    "prompt": prompt
                })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [f"<OBJ{max_id:03}>."],
                "prompt": prompt
            })

    # print(len(new_annos))
    print(f'--- new scanrefer_{segmentor}_{split}{version}_rephrase.json has {len(new_annos)} annotations ---')

    save_path = os.path.join(args.save_dir, f"scanrefer_{segmentor}_{split}{version}_rephrase.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)

# ---- previous scanrefer_mask3d_train.json has 32338 annotations ----
# --- original ScanRefer_filtered_train.json has 36665 annotations ---
# --- new scanrefer_mask3d_train.json has 36187 annotations ---
# percent iou@0.25 of new_annos: 1.0
# percent iou@0.5 of new_annos: 0.9688838533174897

# ---- previous scanrefer_mask3d_val.json has 9508 annotations ----
# --- original ScanRefer_filtered_val.json has 9508 annotations ---
# --- new scanrefer_mask3d_val.json has 9508 annotations ---
# percent iou@0.25 of new_annos: 0.9385780395456458
# percent iou@0.5 of new_annos: 0.8590660496424064

# iou50
# --- new scanrefer_mask3d_train.json has 35061 annotations ---