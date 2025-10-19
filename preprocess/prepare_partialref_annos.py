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
from prompts.prompts import partial_grounding_prompt
import csv
import string
import re
import os


relations =  {
        "supported-by": "on top of/on/lying on",
        "supporting": "supporting",
        "above": "above/over",
        "below": "below/under/beneath/underneath",
        "farthest": "farthest from/far from/far away from",
        "closest": "closer to/close to/near",
        "front": "in front of",
        "right": "on the right of/to the right of/on the right side of",
        "back": "on the back of/behind",
        "left": "on the left of/to the left of/on the left side of"
    }

all_relations = [x for val in relations.values() for x in val.split('/')]

counterpart_types = {
        "farthest": "closest",
        "closest": "farthest",
        "front": "back",
        "back": "front",
        "right": "left",
        "left": "right",
        "above": "below",
        "below": "above",
        "supported-by": "supporting",
        "supporting": "supported-by"
}

def counterpart_relations(caption, relation_type, split):
    if relation_type not in relations:
        return [], [], []
    # longest first
    candidate_relations = sorted(relations[relation_type].split('/'), key=lambda x: len(x), reverse=True)

    found_relation = None
    for current_relation in candidate_relations:
        if current_relation in caption:
            found_relation = current_relation
    
    if found_relation is None:
        return [], [], []
    
    # counterpart_relation = counterpart_types[relation_type]
    # counterpart_candidate_relations = relations[counterpart_relation].split('/')
    counterpart_candidate_relations = [x for x in all_relations if x not in candidate_relations]

    substituted_captions = []
    counterpart_captions = []
    for match in re.finditer(found_relation, caption):
        start_idx = match.start()
        end_idx = match.end()
        for substitute in candidate_relations:
            new_caption = caption[:start_idx] + substitute + caption[end_idx:]
            substituted_captions.append(new_caption)

        for counterpart in counterpart_candidate_relations:
            new_caption = caption[:start_idx] + counterpart + caption[end_idx:]
            counterpart_captions.append(new_caption)

    # 随机抽取20%的counterpart_captions for train
    # 随机抽取50%个counterpart_captions for val
    if split == 'train':
        counterpart_captions = random.sample(counterpart_captions, max([1, int(0.2 * len(counterpart_captions))]))
    else:
        counterpart_captions = random.sample(counterpart_captions, max([1, int(0.5 * len(counterpart_captions))]))
    return counterpart_captions, substituted_captions, candidate_relations


parser = argparse.ArgumentParser()

parser.add_argument('--segmentor', required=True, type=str)
parser.add_argument('--version', type=str, default='')
parser.add_argument('--train_iou_thres', type=float, default=0.75)
parser.add_argument('--save_dir', type=str, default='')
parser.add_argument('--attr_dir', type=str, default='')
args = parser.parse_args()

segmentor = args.segmentor
version = args.version

train_scenes = [x.strip() for x in open('/data/kangweitai/3D/annotation/grounding/meta_data/scannetv2_train.txt').readlines()]
val_scenes = [x.strip() for x in open('/data/kangweitai/3D/annotation/grounding/meta_data/scannetv2_val.txt').readlines()]
scene_lists = {
    'train': train_scenes,
    'val': val_scenes
}

type2class = {'cabinet':0, 'bed':1, 'chair':2, 'sofa':3, 'table':4, 'door':5,
                'window':6,'bookshelf':7,'picture':8, 'counter':9, 'desk':10, 'curtain':11,
                'refrigerator':12, 'shower curtain':13, 'toilet':14, 'sink':15, 'bathtub':16, 'others':17}
class2type = {type2class[t]: t for t in type2class}
raw2label = {}
raw2type = {}
with open('/data/kangweitai/3D/annotation/grounding/meta_data/scannetv2-labels.combined.tsv', 'r') as f:
    csvreader = csv.reader(f, delimiter='\t')
    csvreader.__next__()
    for line in csvreader:
        raw_name = line[1]
        nyu40_name = line[7]
        if nyu40_name not in type2class.keys():
            raw2label[raw_name] = type2class['others']
            raw2type[raw_name] = 'others'
        else:
            raw2label[raw_name] = type2class[nyu40_name]
            raw2type[raw_name] = nyu40_name
            
raw_annos = []
with open('/data/kangweitai/3D/annotation/grounding/ReferIt3D/sr3d+.csv') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        raw_annos.append({
            'scene_id': row['scan_id'],
            'obj_id': int(row['target_id']),
            'description': row['utterance'],
            'object_name': row['instance_type'],
            'relation_type': row['reference_type']
        })

for split in ["train"]: # , "val"
    save_annos = []
    annos = [anno for anno in raw_annos if anno['scene_id'] in scene_lists[split]]
    for anno in annos:
        if anno['description'][-1] in string.punctuation:
            anno['description'] = anno['description'][:-1]

    label2captions = defaultdict(list)
    for anno in annos:
        label = raw2type[anno['object_name']]
        if label != 'others':
            label2captions[label].append(anno['description'])
    

    instance_attribute_file = os.path.join(args.attr_dir, f"scannet_{segmentor}_{split}_attributes{version}.pt")
    scannet_attribute_file = f"/data/kangweitai/3D/chat3d-anno/scannet/scannet_{split}_attributes.pt"
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    unique_scene_object = []
    scene2type = defaultdict(set)
    for scene_id in scannet_attrs:
        object_names = scannet_attrs[scene_id]['objects']
        for name in object_names:
            label = raw2type[name]
            if label != 'others':
                scene2type[scene_id].add(label)

        object_names = [raw2label[name] for name in object_names]
        for objid, name in enumerate(object_names):
            # check each object if it is unique
            if object_names.count(name) == 1 and name != 17:
                unique_scene_object.append(f'{scene_id}|{objid}')

    new_unique_annos = []
    for anno in annos:
        if f"{anno['scene_id']}|{anno['obj_id']}" in unique_scene_object:
            new_unique_annos.append(anno)

    annos = new_unique_annos
    zero_count = 0
    partial_count = 0
    single_count = 0
    for anno in tqdm(annos):
        scene_id = anno['scene_id']
        obj_id = anno['obj_id']
        desc = anno['description']
        if scene_id not in instance_attrs:
            continue
        instance_locs = instance_attrs[scene_id]['locs']
        scannet_locs = scannet_attrs[scene_id]['locs']
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
        if split == 'train':
            if max_iou >= args.train_iou_thres:
                # zero ref:
                if random.random() < 0.5:
                    neg_types = [x for x in list(label2captions.keys()) if x not in scene2type[scene_id]]
                    if len(neg_types) > 0:
                        # neg_type = random.choice(neg_types)
                        if len(neg_types) > 1:
                            neg_types = random.sample(neg_types, 2)
                        for neg_type in neg_types:
                            prompt = random.choice(partial_grounding_prompt).replace('<description>', 
                                                                    random.choice(label2captions[neg_type]))
                            answer = 'No.'
                            save_annos.append({
                                'scene_id': scene_id,
                                'obj_id': 0,
                                'prompt': prompt,
                                'caption': answer
                            })
                            zero_count += 1

                # partial ref:
                counterpart_decs, substitute_decs, correct_relations = \
                    counterpart_relations(desc, anno['relation_type'], split)
                if len(counterpart_decs) == 0:
                    continue
                for choosen_counterpart in counterpart_decs:
                    # prompt = random.choice(partial_grounding_prompt).replace('<description>', 
                    #                         random.choice(counterpart_decs))
                    prompt = random.choice(partial_grounding_prompt).replace('<description>',
                                            choosen_counterpart)
                    for subtitle in correct_relations:
                        answer = f'It is \"{subtitle}\". <OBJ{max_id:03}>.'
                        save_annos.append({
                            'scene_id': scene_id,
                            'obj_id': obj_id,
                            'prompt': prompt,
                            'caption': answer
                        })
                        partial_count += 1

                # single ref:
                if len(substitute_decs) > 0:
                    # if len(substitute_decs) > 1:
                    #     substitute_decs = random.sample(substitute_decs, 2)
                    for dec in substitute_decs:
                        prompt = random.choice(partial_grounding_prompt).replace('<description>', dec)
                        answer = f'Yes. <OBJ{max_id:03}>.'
                        save_annos.append({
                            'scene_id': scene_id,
                            'obj_id': obj_id,
                            'prompt': prompt,
                            'caption': answer
                        })
                        single_count += 1

        else:
            # zero ref:
            # if random.random() < 0.4:
            neg_types = [x for x in list(label2captions.keys()) if x not in scene2type[scene_id]]
            # if len(neg_types) > 0:
            # for choosen_neg_type in neg_types:
            neg_type = random.choice(neg_types)
            prompt = random.choice(partial_grounding_prompt).replace('<description>', 
                                                    random.choice(label2captions[neg_type]))
            save_annos.append({
                'scene_id': scene_id,
                'obj_id': 0,
                'prompt': prompt,
                'ref_captions': ['No.']
            })
            zero_count += 1

            # partial ref:
            counterpart_decs, substitute_decs, correct_relations = \
                counterpart_relations(desc, anno['relation_type'], split)
            if len(counterpart_decs) == 0:
                continue
            for choosen_counterpart in counterpart_decs:
                prompt = random.choice(partial_grounding_prompt).replace('<description>', 
                                        choosen_counterpart)
                answers = [f'It is \"{x}\". <OBJ{max_id:03}>.' for x in correct_relations]
                save_annos.append({
                    'scene_id': scene_id,
                    'obj_id': obj_id,
                    'prompt': prompt,
                    'ref_captions': answers
                })
                partial_count += 1

            # single ref:
            prompt = random.choice(partial_grounding_prompt).replace('<description>', 
                                    random.choice(substitute_decs))
            answer = f'Yes. <OBJ{max_id:03}>.'
            save_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'ref_captions': [answer]
            })
            single_count += 1

    print(f'--- unfactual ref: {zero_count}, partial factual ref: {partial_count}, factual ref: {single_count} ---')
    print(f'--- new partialref_{segmentor}_{split}{version}.json has {len(save_annos)} annotations ---')
    save_path = os.path.join(args.save_dir, f"partialref_{segmentor}_{split}{version}_scale.json")
    with open(save_path, "w") as f:
        json.dump(save_annos, f, indent=4)

# iou=0.50
# --- unfactual ref: 3603, partial factual ref: 16640, factual ref: 3188 ---
# --- new partialref_mask3d_train.json has 23431 annotations ---
# --- unfactual ref: 1800, partial factual ref: 8228, factual ref: 747 ---
# --- new partialref_mask3d_val.json has 10775 annotations ---

# scale
# --- unfactual ref: 7066, partial factual ref: 33352, factual ref: 8338 ---
# --- new partialref_mask3d_train.json has 48756 annotations ---