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

for split in ["train", "val"]:
    new_annos = []

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for scene_id in tqdm(scannet_attrs.keys()):
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]['locs']
        scannet_locs = scannet_attrs[scene_id]['locs']
        segmented_num = len(instance_locs)
        gt_num = len(scannet_locs)
        scannet_class_labels = scannet_attrs[scene_id]['objects']
        for obj_id in range(gt_num):
            class_label = scannet_class_labels[obj_id]
            if any(x in class_label for x in unwanted_words):
                continue
        
            max_iou, max_id = -1, -1
            for pred_id in range(instance_locs.shape[0]):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id
            prompt = f"What is the <OBJ{max_id:03}>?"
            caption = f"<OBJ{max_id:03}> is a {class_label}."
            if split == 'train':
                if max_iou >= args.train_iou_thres:
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': obj_id,
                        'prompt': prompt,
                        'caption': caption
                    })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': prompt,
                    'ref_captions': [caption] 
                })
                
    save_path = os.path.join(args.save_dir, f"obj_align_{segmentor}_{split}{version}.json")
    print(f'--- new obj_align_{segmentor}_{split}{version}.json has {len(new_annos)} annotations ---')
    with open(save_path, 'w') as f:
        json.dump(new_annos, f, indent=4)

# ---- previous obj_align_mask3d_train.json has 23333 annotations ----
# --- new obj_align_mask3d_train.json has 74327 annotations ---
# ---- previous obj_align_mask3d_val.json has 7839 annotations ----
# --- new obj_align_mask3d_val.json has 17951 annotations ---

# --- new obj_align_mask3d_train.json has 27375 annotations ---

#iou50
# --- new obj_align_mask3d_train.json has 26010 annotations ---