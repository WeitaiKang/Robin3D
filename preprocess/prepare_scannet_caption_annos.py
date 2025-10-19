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
from prompts.prompts import obj_caption_wid_prompt
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

for split in ['train']:
    annos = json.load(open(f"/data/kangweitai/3D/annotation/grounded_scene_caption/scannet_{split}_caption.json"))
    new_annos = []

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for i, anno in tqdm(enumerate(annos), total=len(annos)):
        scene_id = anno['scene_id']
        obj_id = anno['obj_id']
    
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
        if split == 'train':
            if max_iou > args.train_iou_thres:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{max_id:03}>"),
                    'caption': anno['caption']
                })
        else:
            new_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': random.choice(obj_caption_wid_prompt).replace('<id>', f"<OBJ{max_id:03}>"),
                'ref_captions': [ref for ref in anno['ref_captions']] 
            })
    
    print(f"scannet_caption_{segmentor}_{split}{version}.json, {len(annos)} -> {len(new_annos)}")

    save_path = os.path.join(args.save_dir, f"scannet_caption_{segmentor}_{split}{version}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)
