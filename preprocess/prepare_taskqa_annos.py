import json
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
import argparse
from utils.box_utils import box3d_iou, construct_bbox_corners
parser = argparse.ArgumentParser()
import os
import string

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

neg_answers = [
    "No object can fulfill the requirement.",
    "No object can satisfy the requirement.",
    "No object can meet the task.",
    "No object can meet the requirement.",
    "No object can meet the demand.",
    "No object can meet the request.",
]

for split in ['train', 'val']:
    with open(f"/data/kangweitai/3D/annotation/taskqa/scene_plan_{split}_gpt4v3_all_extend_withZero_duplicate.json", "r") as f:
        annos = json.load(f)
    print(len(annos))
    new_annos = []
    positive_sample_num = 0
    negative_sample_num = 0

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    for i, anno in tqdm(enumerate(annos), total=len(annos)):
        scene_id = anno['scene_id']
    
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        gt_ids = [] 
        for x in anno['object_id']:
            assert len(x) == 1
            x = [int(y) for y in x]
            gt_ids.extend(x)
        prompt = anno['question'] + " And please provide the ID for the object related to your answer."
        assert len(gt_ids) == len(anno['answer'])
    
        if len(gt_ids) == 0:
            negative_sample_num += 1
            if split == "train":
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": 0,
                    "caption": "No.",
                    "prompt": prompt
                })
            else:
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": 0,
                    "ref_captions": ["No."],
                    "prompt": prompt
                })
            continue

        answers = []
        max_iou, max_id = -1, -1
        for i, obj_id in enumerate(gt_ids):
            for pred_id in range(instance_num):
                pred_locs = instance_locs[pred_id].tolist()
                gt_locs = scannet_locs[obj_id].tolist()
                pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = pred_id
            answer = anno['answer'][i].capitalize()
            if answer[-1] in string.punctuation:
                answer = answer[:-1]
                answer = answer + "."
            answer = answer + " " + f"<OBJ{max_id:03}>."
            answer = "Yes. " + answer
            answers.append(answer)
            
            if split == "train" and max_iou >= args.train_iou_thres:
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": obj_id,
                    "caption": answer,
                    "prompt": prompt
                })
                positive_sample_num += 1

        if split != "train": 
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_id,
                "ref_captions": [answer for answer in answers],
                "prompt": prompt
            })
            positive_sample_num += 1

    print(f'--- save {len(new_annos)} annotations with {negative_sample_num} neg & {positive_sample_num} pos in {split} ---')
    save_path = os.path.join(args.save_dir, f"taskqa_{segmentor}_{split}{version}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)

#iou=0.25
# --- save 23129 annotations with 1411 neg & 21718 pos in train ---
# --- save 2896 annotations with 370 neg & 2526 pos in val ---

#iou=0.5
# --- save 22805 annotations with 1411 neg & 21394 pos in train ---
# --- save 2896 annotations with 370 neg & 2526 pos in val ---