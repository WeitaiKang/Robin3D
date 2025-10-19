import json
import numpy as np
import os
import nltk
import random
from tqdm import tqdm
import sys
sys.path.append('.')
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='')
args = parser.parse_args()

anno_dir = '/data/kangweitai/3D/annotation/sqa3d/balanced/'


def convert_person_view(sentence):
    # first-person view to second-person view
    forms = {'i': 'you', 'me': 'you', 'my': 'your', 'mine': 'yours', 'am': 'are'}
    def translate(word):
        if word.lower() in forms:
            return forms[word.lower()]
        return word
    result = ' '.join([translate(word) for word in nltk.wordpunct_tokenize(sentence)])
    return result.capitalize()


def get_sqa_question_type(question):
    question = question.lstrip()
    if question[:4].lower() == 'what':
        return 0
    elif question[:2].lower() == 'is':
        return 1
    elif question[:3].lower() == 'how':
        return 2
    elif question[:3].lower() == 'can':
        return 3
    elif question[:5].lower() == 'which':
        return 4
    else:
        return 5   # others


for split in ['test']: # 'train', 'val'
    scan_ids = []
    sqa_annos = []
    question_file = os.path.join(anno_dir, f'v1_balanced_questions_{split}_scannetv2.json')
    with open(question_file, 'r', encoding='utf-8') as f:
        question_data = json.load(f)['questions']
    question_map = {}
    for item in question_data:
        question_map[item['question_id']] = {
            's': [item['situation']] + item['alternative_situation'],   # list of str
            'q': item['question'],   # str
        }

    anno_file = os.path.join(anno_dir, f'v1_balanced_sqa_annotations_{split}_scannetv2.json')
    with open(anno_file, 'r', encoding='utf-8') as f:
        anno_data = json.load(f)['annotations']
    for item in tqdm(anno_data):
        scan_ids.append(item['scene_id'])
        scene_id = item['scene_id']
        obj_id = 0
        situation = random.choice(question_map[item['question_id']]['s'])
        question = question_map[item['question_id']]['q']
        question_type = get_sqa_question_type(question)
        prompt = situation + ' ' + question + " Answer the question using a word or a phrase."
        answers = [meta['answer'] for meta in item['answers']]
        if split == 'train':
            answer = random.choice(answers)
            answer = answer.capitalize()
            if answer[-1] != ".":
                answer += "."
            sqa_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'caption': answer,
                'sqa_type': question_type
            })
        else:
            sqa_annos.append({
                'scene_id': scene_id,
                'obj_id': obj_id,
                'prompt': prompt,
                'ref_captions': [answer for answer in answers], 
                'sqa_type': question_type
            })
    
    print(f'--- sqa3d_{split}.json has {len(sqa_annos)} annotations ---')
    save_path = os.path.join(args.save_dir, f"sqa3d_{split}.json")
    with open(save_path, "w") as f:
        json.dump(sqa_annos, f, indent=4)

# ---- previous sqa3d_train.json has 26623 annotations ----
# ---- previous sqa3d_val.json has 3261 annotations ----
    

