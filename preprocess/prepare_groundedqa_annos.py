import json
import sys
sys.path.append('.')
import torch
from tqdm import tqdm
import argparse
from utils.box_utils import box3d_iou, construct_bbox_corners
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

for split in ['train', 'val']:
    # with open(f"/data/kangweitai/3D/chat3d-anno/chat3d-anno-iou50-oneprompt-0.3/groundedqa_mask3d_{split}.json", "r") as f:
    #     annos = json.load(f)
    # for anno in annos:
    #     anno["prompt"] = anno["prompt"].replace("provide the IDs for objects related to", 
    #                                             "provide all the IDs for objects related to")
    # save_path = os.path.join(args.save_dir, f"groundedqa_{segmentor}_{split}{version}.json")
    # with open(save_path, "w") as f:
    #     json.dump(annos, f, indent=4)
    # continue

    with open(f"/data/kangweitai/3D/annotation/ScanQA/General_ScanQA_v1.0_{split}.json", "r") as f:
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
        gt_ids = anno['object_ids']
        question = anno['question']
        prompt = question + " If you can, answer the question using a word or a phrase. And provide all the IDs for objects related to the question and answer."
        
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
        pred_ids = sorted(pred_ids)
        pred_id_strs = ["<OBJ{:03}>".format(pred_id) for pred_id in pred_ids]

        if len(pred_ids) == 1:
            pred_id_strs = f"{pred_id_strs[0]}."
        elif len(pred_ids) == 2:
            pred_id_strs = f"{pred_id_strs[0]} and {pred_id_strs[1]}."
        elif len(pred_ids) > 2:
            pred_id_strs = f"{', '.join(pred_id_strs[:-1])}, and {pred_id_strs[-1]}."

        if len(gt_ids) == 0:
            # answers = [a.capitalize() for a in anno['answers']]
            answers = ["No."]
            negative_sample_num += 1
        else:
            answers = [f"{a.capitalize()}. {pred_id_strs}" for a in anno['answers']]
            if split == 'train':
                positive_sample_num += len(answers)
            else:
                positive_sample_num += 1

        if split == 'train':
            for i in range(len(answers)):
                if i > 0 and answers[i] == answers[i-1]:
                    continue
                answer = answers[i]
                if answer[-1] != ".":
                    answer += "."
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": gt_ids,
                    "prompt": prompt,
                    "caption": answer,
                })
        else:
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": gt_ids,
                "prompt": prompt,
                "ref_captions": [answer for answer in answers] 
                
            })

    save_path = os.path.join(args.save_dir, f"groundedqa_{segmentor}_{split}{version}.json")
    print(f'--- save {save_path} {len(new_annos)} annotations with {negative_sample_num} neg & {positive_sample_num} pos in {split} ---')
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)

#iou=0.25
# --- save 30702 annotations with 4996 neg & 26077 pos in train ---
# --- save 5671 annotations with 996 neg & 4675 pos in val ---

#iou=0.5
# --- save 29773 annotations with 4996 neg & 25131 pos in train ---
# --- save 5671 annotations with 996 neg & 4675 pos in val ---