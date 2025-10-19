import json
import sys
sys.path.append('.')
import os

import argparse
parser = argparse.ArgumentParser()
parser.add_argument('--save_dir', type=str, default='')
args = parser.parse_args()

for split in ['train', 'val']:
    with open(f"/data/kangweitai/3D/annotation/ScanQA/ScanQA_v1.0_{split}.json", "r") as f:
        annos = json.load(f)
    print(len(annos))
    new_annos = []
    for anno in annos:
        scene_id = anno["scene_id"]
        obj_ids = anno["object_ids"] if "object_ids" in anno else []
        question = anno["question"]

        prompt = question + " Answer the question using a word or a phrase."

        answers = anno["answers"]
        if split == "val":
            new_annos.append({
                "scene_id": scene_id,
                "obj_id": obj_ids[0],
                "prompt": prompt,
                "ref_captions": [answer for answer in answers], 
            })
        elif split == "train":
            for i in range(len(answers)):
                if i > 0 and answers[i] == answers[i-1]:
                    continue
                answer = answers[i]
                answer = answer.capitalize()
                if answer[-1] != ".":
                    answer += "."
                new_annos.append({
                    "scene_id": scene_id,
                    "obj_id": obj_ids[0],
                    "prompt": prompt,
                    "caption": answer,
                })
    print(f'--- "scanqa_{split}.json"', len(new_annos))
    save_path = os.path.join(args.save_dir, f"scanqa_{split}.json")
    with open(save_path, "w") as f:
        json.dump(new_annos, f, indent=4)

# ---- previous scanqa_train.json has 26138 annotations ----
# ---- previous scanqa_val.json has 4675 annotations ----