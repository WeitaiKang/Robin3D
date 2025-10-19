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
from prompts.prompts import scan2cap_prompt
import nltk
import os


def capitalize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    result = ' '.join(capitalized_sentences)
    return result


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()


for split in ["train", "val"]:
    segmentor = args.segmentor
    version = args.version
    annos = json.load(open(f"/data/kangweitai/3D/annotation/grounding/ScanRefer/ScanRefer_filtered_{split}.json", "r"))
    new_annos = []
    print(f'--- original ScanRefer_filtered_{split}.json has {len(annos)} annotations ---')

    scene_ids = set()
    corpus = defaultdict(list)
    for anno in annos:
        gt_key = f"{anno['scene_id']}|{anno['object_id']}"
        description = capitalize_sentences(anno['description'])
        corpus[gt_key].append(description)
        scene_ids.add(anno['scene_id'])
    scene_ids = list(scene_ids)

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')


    covered25_num, covered50_num = 0, 0
    count_all = 0
    total_gt_bbox = 0
    cap_gt_bbox = 0
    for scene_id in tqdm(scene_ids, total=len(scene_ids)):
        total_gt_bbox += len(scannet_attrs[scene_id]["locs"])
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        segmented_num = len(instance_locs)
        gt_num = len(scannet_locs)
        gt_match_id = [-1] * gt_num
        gt_match_iou = [-1] * gt_num
        for pred_id in range(segmented_num):
        
            pred_locs = instance_locs[pred_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            max_id = max_iou = -1
            for gt_id in range(len(scannet_locs)):
                if f"{scene_id}|{gt_id}" not in corpus:
                    continue
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = gt_id
            if f"{scene_id}|{max_id}" not in corpus:
                continue
            if max_iou > gt_match_iou[max_id]:
                gt_match_iou[max_id] = max_iou
                gt_match_id[max_id] = pred_id
        for gt_id, pred_id in enumerate(gt_match_id):
            if f"{scene_id}|{gt_id}" in corpus:
                count_all += len(corpus[f"{scene_id}|{gt_id}"])
            if pred_id == -1:
                continue
            if split == 'train' and gt_match_iou[gt_id] < args.train_iou_thres:
                continue
            if gt_match_iou[gt_id] >= 0.25:
                covered25_num += len(corpus[f"{scene_id}|{gt_id}"])
            if gt_match_iou[gt_id] >= 0.5:
                covered50_num += len(corpus[f"{scene_id}|{gt_id}"])
            if split == 'train':
                for caption in corpus[f"{scene_id}|{gt_id}"]:
                    new_annos.append({
                        'scene_id': scene_id,
                        'obj_id': gt_id,
                        'pred_id': pred_id,
                        'prompt': random.choice(scan2cap_prompt).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                        "caption": caption,
                        "iou": gt_match_iou[gt_id]
                    })
            else:
                new_annos.append({
                    'scene_id': scene_id,
                    'obj_id': gt_id,
                    'pred_id': pred_id,
                    'prompt': random.choice(scan2cap_prompt).replace(f"<id>", f"<OBJ{pred_id:03}>"),
                    "ref_captions": [caption for caption in corpus[f"{scene_id}|{gt_id}"]],
                    "iou": gt_match_iou[gt_id]
                })

    print(f'--- annotated bbox: {len(list(corpus.keys()))}, scene gtbbox {total_gt_bbox} ---')
    print(f'--- new scan2cap_{segmentor}_{split}{version}.json has {len(new_annos)} annotations ---')
    print(f"percent iou@0.25 of new_annos: {covered25_num / len(annos)}")
    print(f"percent iou@0.5 of new_annos: {covered50_num / len(annos)}")

    save_path = os.path.join(args.save_dir, f"scan2cap_{segmentor}_{split}{version}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)

# --- original ScanRefer_filtered_train.json has 36665 annotations ---
# ---- previous scan2cap_mask3d_train.json has 32338 annotations ----
# --- annotated scene-objid: 7875, scene gtbbox 17211 ---
# --- new scan2cap_mask3d_train.json has 36079 annotations ---
# percent iou@0.25 of new_annos: 0.9840174553388791
# percent iou@0.5 of new_annos: 0.9561161870994136

# --- original ScanRefer_filtered_val.json has 9508 annotations ---
# ---- previous scan2cap_mask3d_val.json has 2007 scene-objid ----
# --- annotated scene-objid: 2068, scene gtbbox 4593 ---
# --- new scan2cap_mask3d_val.json has 2007 scene-objid ---
# percent iou@0.25 of new_annos: 0.92406394615061
# percent iou@0.5 of new_annos: 0.8574884307951199

#iou50
# --- new scan2cap_mask3d_train.json has 35056 annotations ---