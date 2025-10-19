from plyfile import PlyData
import numpy as np
import os
import json
import torch
from collections import defaultdict
from tqdm import tqdm
import argparse
import nltk
import random
import sys
sys.path.append('.')
from utils.box_utils import get_box3d_min_max, box3d_iou, construct_bbox_corners
from prompts.prompts import pointedcap_prompt
import string
parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

raw_data_dir = "/data/kangweitai/3D/scannet/scans"

def capitalize_sentences(text):
    sentences = nltk.sent_tokenize(text)
    capitalized_sentences = [sentence.capitalize() for sentence in sentences]
    result = ' '.join(capitalized_sentences)
    return result

neg_answers = [
    "No object is clicked.",
    "No object is selected.",
    "No item has been chosen.",
    "No object has been clicked.",
    "No item is clicked.",
    "No object has been chosen.",
    "No item is selected."
]

for split in ["train"]: # , "val"
    new_annos= []
    segmentor = args.segmentor
    version = args.version
    split_file = f"/data/kangweitai/3D/annotation/grounding/meta_data/scannetv2_{split}.txt"
    scan_names = [line.rstrip() for line in open(split_file)]
    print(f'{split} split scans: {len(scan_names)}')

    cap_annos = json.load(open(f"/data/kangweitai/3D/annotation/grounding/ScanRefer/ScanRefer_filtered_{split}.json", "r"))
    cap_scene_ids = set()
    corpus = defaultdict(list)
    for anno in cap_annos:
        gt_key = f"{anno['scene_id']}|{anno['object_id']}"
        description = capitalize_sentences(anno['description'])
        # if description in string.punctuation:
        #     description = description[:-1]
        corpus[gt_key].append(description)
        cap_scene_ids.add(anno['scene_id'])
    cap_scene_ids = list(cap_scene_ids)

    
    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    total_positive_num = 0
    total_negative_click_num = 0
    total_negative_category_num = 0
    for scan_id in tqdm(scan_names):
        
        if scan_id not in cap_scene_ids:
            continue
        if scan_id not in instance_attrs:
            continue

        # prepare scene attributes
        aggregation_path = os.path.join(raw_data_dir, scan_id, scan_id + '.aggregation.json')
        segs_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.0.010000.segs.json')
        scan_ply_path = os.path.join(raw_data_dir, scan_id, scan_id + '_vh_clean_2.labels.ply')

        data = PlyData.read(scan_ply_path)
        x = np.asarray(data.elements[0].data['x']).astype(np.float32)
        y = np.asarray(data.elements[0].data['y']).astype(np.float32)
        z = np.asarray(data.elements[0].data['z']).astype(np.float32)
        pc = np.stack([x, y, z], axis=1)

        align_matrix = np.eye(4)
        with open(os.path.join(raw_data_dir, scan_id, '%s.txt'%(scan_id)), 'r') as f:
            for line in f:
                if line.startswith('axisAlignment'):
                    align_matrix = np.array([float(x) for x in line.strip().split()[-16:]]).astype(np.float32).reshape(4, 4)
                    break

        pts = np.ones((pc.shape[0], 4), dtype=pc.dtype)
        pts[:, 0:3] = pc
        pc = np.dot(pts, align_matrix.transpose())[:, :3]

        scan_aggregation = json.load(open(aggregation_path))
        segments_info = json.load(open(segs_path))
        segment_indices = segments_info["segIndices"]
        segment_indices_dict = defaultdict(list)
        for i, s in enumerate(segment_indices):
            segment_indices_dict[s].append(i)
        
        instance_labels = []
        inst_locs = []
        instance_points = []
        instance_centers = []
        rest_points_idx = [ _ for _ in range(pc.shape[0])]
        for idx, object_info in enumerate(scan_aggregation['segGroups']):
            object_instance_label = object_info['label']
            object_id = object_info['objectId']
            segments = object_info["segments"]
            pc_ids = []
            for s in segments:
                pc_ids.extend(segment_indices_dict[s])
            object_pc = pc[pc_ids]
            object_center = (np.max(object_pc, axis=0) + np.min(object_pc, axis=0)) / 2.0
            object_size = np.max(object_pc, axis=0) - np.min(object_pc, axis=0)
            object_bbox = torch.from_numpy(np.concatenate([object_center, object_size], axis=0))
            inst_locs.append(object_bbox)
            instance_labels.append(object_instance_label)
            instance_points.append(object_pc)
            instance_centers.append(object_center)
            rest_points_idx = list(set(rest_points_idx) - set(pc_ids))

        inst_locs = torch.stack(inst_locs, dim=0)
        current_point_files = {
            'objects': instance_labels,
            'points': instance_points,
            'centers': instance_centers,
            'rest_points': pc[rest_points_idx],
            'min': np.min(pc, axis=0),
            'max': np.max(pc, axis=0),
        }

        # prepare annotations
        instance_locs = instance_attrs[scan_id]["locs"]
        scannet_locs = scannet_attrs[scan_id]["locs"]
        segmented_num = len(instance_locs)
        gt_num = len(scannet_locs)
        gt_match_id = [-1] * gt_num
        gt_match_iou = [-1] * gt_num
        for pred_id in range(segmented_num):
        
            pred_locs = instance_locs[pred_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            max_id = max_iou = -1
            for gt_id in range(len(scannet_locs)):
                if f"{scan_id}|{gt_id}" not in corpus:
                    continue
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
                iou = box3d_iou(pred_corners, gt_corners)
                if iou > max_iou:
                    max_iou = iou
                    max_id = gt_id
            if f"{scan_id}|{max_id}" not in corpus:
                continue
            if max_iou > gt_match_iou[max_id]:
                gt_match_iou[max_id] = max_iou
                gt_match_id[max_id] = pred_id

        all_categories = set(current_point_files['objects'])
        for gt_id, pred_id in enumerate(gt_match_id):
            if f"{scan_id}|{gt_id}" not in corpus:
                continue
            if pred_id == -1:
                pred_id = random.randint(0, segmented_num - 1)
            if split == 'train':
                # add positive click
                if gt_match_iou[gt_id] >= args.train_iou_thres and len(current_point_files['points'][gt_id]) > 0:
                    for caption in corpus[f"{scan_id}|{gt_id}"]:
                        new_annos.append({
                            'scene_id': scan_id,
                            'obj_id': gt_id,
                            'pred_id': pred_id,
                            'prompt': random.choice(pointedcap_prompt), #.replace('<category>', current_point_files['objects'][gt_id]),
                            "caption": caption + f' <OBJ{pred_id:03}>.',
                            "iou": gt_match_iou[gt_id],
                            'click': current_point_files['centers'][gt_id].tolist(),
                            'min': current_point_files['min'].tolist(),
                            'max': current_point_files['max'].tolist()
                        })
                        total_positive_num += 1
                        new_annos.append({
                            'scene_id': scan_id,
                            'obj_id': gt_id,
                            'pred_id': pred_id,
                            'prompt': random.choice(pointedcap_prompt), #.replace('<category>', current_point_files['objects'][gt_id]),
                            "caption": caption + f' <OBJ{pred_id:03}>.',
                            "iou": gt_match_iou[gt_id],
                            'click': random.choice(current_point_files['points'][gt_id]).tolist(),
                            'min': current_point_files['min'].tolist(),
                            'max': current_point_files['max'].tolist()
                        })
                        total_positive_num += 1
                # add negative click
                if random.random() < 0.5 and len(current_point_files['rest_points']) > 0:
                    for _ in range(2):
                        new_annos.append({
                            'scene_id': scan_id,
                            'obj_id': gt_id,
                            'pred_id': pred_id,
                            'prompt': random.choice(pointedcap_prompt), #.replace('<category>', current_point_files['objects'][gt_id]),
                            # "caption": random.choice(neg_answers),
                            "caption": "No.",
                            "iou": gt_match_iou[gt_id],
                            'click': random.choice(current_point_files['rest_points']).tolist(),
                            'min': current_point_files['min'].tolist(),
                            'max': current_point_files['max'].tolist()
                        })
                        total_negative_click_num += 1
                # add negative category
                # neg_category = random.choice(
                #     list(all_categories - set([current_point_files['objects'][gt_id]]))
                # )
                # neg_answer = f"No, this is not a {neg_category}, but rather a {current_point_files['objects'][gt_id]}."
                # if random.random() < 0.3:
                #     new_annos.append({
                #         'scene_id': scan_id,
                #         'obj_id': gt_id,
                #         'pred_id': pred_id,
                #         'prompt': random.choice(pointedcap_prompt).replace('<category>', neg_category),
                #         "caption": neg_answer,
                #         "iou": gt_match_iou[gt_id],
                #         'click': random.choice(current_point_files['points'][gt_id]).tolist(),
                #         'min': current_point_files['min'].tolist(),
                #         'max': current_point_files['max'].tolist()
                #     })
                #     total_negative_category_num += 1
            else:
                # add positive click
                if len(current_point_files['points'][gt_id]) > 0:
                    new_annos.append({
                        'scene_id': scan_id,
                        'obj_id': gt_id,
                        'pred_id': pred_id,
                        'prompt': random.choice(pointedcap_prompt), #.replace('<category>', current_point_files['objects'][gt_id]),
                        "ref_captions": [caption + f' <OBJ{pred_id:03}>.' for caption in corpus[f"{scan_id}|{gt_id}"]],
                        "iou": gt_match_iou[gt_id],
                        # 'click': random.choice(current_point_files['points'][gt_id]).tolist(),
                        'click': current_point_files['centers'][gt_id].tolist(),
                        'min': current_point_files['min'].tolist(),
                        'max': current_point_files['max'].tolist()
                    })
                    total_positive_num += 1
                # add negative click
                if len(current_point_files['rest_points']) > 0:
                    new_annos.append({
                        'scene_id': scan_id,
                        'obj_id': gt_id,
                        'pred_id': pred_id,
                        'prompt': random.choice(pointedcap_prompt), #.replace('<category>', current_point_files['objects'][gt_id]),
                        # "ref_captions": neg_answers,
                        "ref_captions": ["No."],
                        "iou": gt_match_iou[gt_id],
                        'click': random.choice(current_point_files['rest_points']).tolist(),
                        'min': current_point_files['min'].tolist(),
                        'max': current_point_files['max'].tolist()
                    })
                    total_negative_click_num += 1
                # add negative category
                # neg_category = random.choice(
                #     list(all_categories - set([current_point_files['objects'][gt_id]]))
                # )
                # neg_answer = f"No, this is not a {neg_category}, but rather a {current_point_files['objects'][gt_id]}."
                # new_annos.append({
                #     'scene_id': scan_id,
                #     'obj_id': gt_id,
                #     'pred_id': pred_id,
                #     'prompt': random.choice(pointedcap_prompt).replace('<category>', neg_category),
                #     "ref_captions": [neg_answer],
                #     "iou": gt_match_iou[gt_id],
                #     'click': random.choice(current_point_files['points'][gt_id]).tolist(),
                #     'min': current_point_files['min'].tolist(),
                #     'max': current_point_files['max'].tolist()
                # })
                total_negative_category_num += 1
        # if len(new_annos) > 5: break
    print(f'--- save data num: {len(new_annos)} to pointedcap_{segmentor}_{split}{version}.json --- total positive num: {total_positive_num}, total negative click num: {total_negative_click_num} -- total negative category num: {total_negative_category_num} --')
    save_path = os.path.join(args.save_dir, f"pointedcap_{segmentor}_{split}{version}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)


