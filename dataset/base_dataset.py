import logging
import os
import random
from torch.utils.data import Dataset
import torch
import glob
from torch.nn.utils.rnn import pad_sequence
import re
import numpy as np
from utils.box_utils import get_3d_box_batch_lwh_tensor, box3d_iou_batch_tensor
from transformers import LlamaTokenizer
from collections import OrderedDict

logger = logging.getLogger(__name__)


neg_qa = [
    "No related object in the scene.",
    "The scene lacks a related object.",
    "No relevant object in this scene.",
    "Scene has no related object.",
    "No object linked to the question.",
    "Scene is missing the relevant object.",
    "No object in the scene fits the question.",
    "No relevant object here.",
    "Scene contains no related object.",
    "The object isn't in this scene."
]

neg_cap = [
    "No object where you click.",
    "Nothing at the click spot.",
    "No object at your click.",
    "Clicked spot is empty.",
    "Nothing found where you clicked.",
    "No object at the click location.",
    "No item where you clicked.",
    "Clicked area has nothing.",
    "No object on your click.",
    "The click spot has no object."
]

class BaseDataset(Dataset):

    def __init__(self):
        self.media_type = "point_cloud"
        self.anno = None
        self.attributes = None
        self.gt_attributes = None
        self.mask3d_feat_pos = None
        self.mask3d_feats = None
        self.feats = None
        self.img_feats = None
        self.scene_mask3d_feat_poses = None
        self.scene_mask3d_feats = None
        self.scene_feats = None
        self.scene_img_feats = None
        self.scene_masks = None
        self.feat_dim = 1024
        self.img_feat_dim = 1536
        self.max_obj_num = 100
        self.scene_obj_labels = None
        self.system="A chat between a curious user and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the user's questions."
        self.instruction="The conversation centers around an indoor scene, with the following object information:"
        self.role=("USER", "ASSISTANT")
        self.start_sym="<s>"
        self.end_sym="</s>"
        
        # 修改
        # /home/weitai/3DLLM/vicuna-7b-v1.5/
        # /data/kangweitai/LLM/vicuna-7b-v1.5/
        self.llama_tokenizer = LlamaTokenizer.from_pretrained("/data/kangweitai/LLM/vicuna-7b-v1.5/", use_fast=False, legacy=False)
        objid_tokens = []
        for i in range(150):
            objid_tokens.append(f"<OBJ{i:03}>")
        self.llama_tokenizer.add_tokens(objid_tokens, special_tokens=True)
        # self.llama_tokenizer.add_tokens(['<click>'], special_tokens=True)
        self.num_expanded = 3 + 1

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        raise NotImplementedError
    
    def prepare_scene_features(self):
        if self.feats is not None:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.feats.keys())
        else:
            scan_ids = set('_'.join(x.split('_')[:2]) for x in self.img_feats.keys())
        scene_mask3d_feat_poses = {}
        scene_mask3d_feats = {}
        scene_feats = {}
        scene_img_feats = {}
        for scan_id in scan_ids:
            if scan_id not in self.attributes:
                continue
            scene_attr = self.attributes[scan_id]
            obj_num = scene_attr['locs'].shape[0]
            obj_ids = scene_attr['obj_ids'] if 'obj_ids' in scene_attr else [_ for _ in range(obj_num)]
            scene_mask3d_feat_pos = []
            scene_mask3d_feat = []
            scene_feat = []
            scene_img_feat = []
            for _i, _id in enumerate(obj_ids):
                item_id = '_'.join([scan_id, f'{_id:02}'])
                if self.mask3d_feat_pos is None or item_id not in self.mask3d_feat_pos:
                    scene_mask3d_feat_pos.append(torch.zeros(128))
                else:
                    scene_mask3d_feat_pos.append(self.mask3d_feat_pos[item_id])
                if self.mask3d_feats is None or item_id not in self.mask3d_feats:
                    scene_mask3d_feat.append(torch.zeros(128))
                else:
                    scene_mask3d_feat.append(self.mask3d_feats[item_id])
                if self.feats is None or item_id not in self.feats:
                    scene_feat.append(torch.zeros(self.feat_dim))
                else:
                    scene_feat.append(self.feats[item_id])
                if self.img_feats is None or item_id not in self.img_feats:
                    scene_img_feat.append(torch.zeros(self.img_feat_dim))
                else:
                    scene_img_feat.append(self.img_feats[item_id].float())
                            
            scene_mask3d_feat_poses[scan_id] = torch.stack(scene_mask3d_feat_pos, dim=0)
            scene_mask3d_feats[scan_id] = torch.stack(scene_mask3d_feat, dim=0)
            scene_feat = torch.stack(scene_feat, dim=0)
            scene_img_feat = torch.stack(scene_img_feat, dim=0)
            scene_feats[scan_id] = scene_feat
            scene_img_feats[scan_id] = scene_img_feat
        return scene_mask3d_feat_poses, scene_mask3d_feats, scene_feats, scene_img_feats

    def get_anno(self, index):
        scene_id = self.anno[index]["scene_id"]
        scene_feat = self.scene_feats[scene_id]
        scene_img_feat = self.scene_img_feats[scene_id]
        scene_mask3d_feat = self.scene_mask3d_feats[scene_id]
        scene_mask3d_feat_pos = self.scene_mask3d_feat_poses[scene_id]

        return scene_mask3d_feat_pos, scene_mask3d_feat, scene_feat, scene_img_feat, scene_id
    
    def get_tgt_mask(self, answer):
        tgt_mask = torch.zeros(150)
        for match in re.finditer("<OBJ\\d{3}>", answer):
            idx = match.start()
            tgt_mask[int(answer[idx+4:idx+7])] = 1
        return tgt_mask
    
    def get_token_ids_mask(self, question, caption=None, eval=False):
        if self.dataset_name in ['scanqa', 'sqa3d','scanqav2', 'sqa3dv2', 'sqa3d_test']:
            question.replace('Answer the question using a word or a phrase.', 'Please provide a brief answer.')
        if self.dataset_name in ['groundedqa', 'groundedqav2']:
            question.replace('answer the question using a word or a phrase.', 'please provide a brief answer.')

        s_q1 = self.system + f' {self.role[0]}: {question} Here are the objects in the scene: '
        # s_q1 = self.system + f' Here are the objects in the scene: '
        s_q1_token = self.llama_tokenizer(s_q1, return_tensors="pt", add_special_tokens=True)
        s_q1_token_token_id = s_q1_token['input_ids'][0]
        s_q1_token_token_mask = s_q1_token['attention_mask'][0]

        s_q2 = f' {self.role[1]}: '
        # s_q2 = f' {self.role[0]}: {question} {self.role[1]}: '
        s_q2_token = self.llama_tokenizer(s_q2, return_tensors="pt", add_special_tokens=False)
        s_q2_token_token_id = s_q2_token['input_ids'][0]
        s_q2_token_token_mask = s_q2_token['attention_mask'][0]
        
        all_token_id = torch.cat([s_q1_token_token_id, torch.zeros(150*self.num_expanded), s_q2_token_token_id])
        all_token_mask = torch.cat([s_q1_token_token_mask, torch.ones(150*self.num_expanded), s_q2_token_token_mask])
        tgt_idx = torch.ones_like(all_token_mask) * -100
        obj_mask = torch.cat([torch.zeros(len(s_q1_token_token_id)), 
                              torch.ones(150*self.num_expanded), 
                              torch.zeros(len(s_q2_token_token_id))])  
        
        if caption is not None:
            a = caption + self.end_sym
            a_token = self.llama_tokenizer(a, return_tensors="pt", add_special_tokens=False)
            a_token_id = a_token['input_ids'][0]
            a_token_mask = a_token['attention_mask'][0]

            all_token_id = torch.cat([all_token_id, a_token_id])
            all_token_mask = torch.cat([all_token_mask, a_token_mask])
            tgt_idx = torch.cat([tgt_idx, a_token_id])
            obj_mask = torch.cat([obj_mask, torch.zeros(len(a_token_id))])
        
#         --- all_token_id: torch.Size([689])                                                                                                                                                        
# --- all_token_mask: torch.Size([689])                                                                                                                                                      
# --- tgt_idx: torch.Size([689])                                                                                                                                                             
# --- obj_mask: torch.Size([689])    
        # max_len = 976
        # if not eval:
        #     if len(all_token_id) > max_len:
        #         all_token_id = torch.cat([all_token_id[:max_len-1], all_token_id[[-1]]])
        #         all_token_mask = torch.cat([all_token_mask[:max_len-1], all_token_mask[[-1]]])
        #         tgt_idx = torch.cat([tgt_idx[:max_len-1], tgt_idx[[-1]]])
        #         obj_mask = torch.cat([obj_mask[:max_len-1], obj_mask[[-1]]])

        return {
            "all_token_id": all_token_id,
            "all_token_mask": all_token_mask,
            "tgt_idx": tgt_idx,
            "obj_mask": obj_mask,
        }

    def get_special_str_index(self, special_str, all_token_id, for_hidden=False):
        target_id = self.llama_tokenizer.convert_tokens_to_ids(special_str)
        target_index = torch.zeros(len(all_token_id))
        index_value = 1
        for i, token_id in enumerate(all_token_id):
            if token_id == target_id:
                if for_hidden:
                    record_idx = i-1
                else:
                    record_idx = i
                target_index[record_idx] = index_value
                index_value += 1

        return target_index
    
    def get_valid_objects_index(self, scene_id):
        gt_locs = self.gt_attributes[scene_id]["locs"]
        pred_locs = self.attributes[scene_id]["locs"]
        num_gts = gt_locs.shape[0]
        num_preds = pred_locs.shape[0]

        pred_corners = get_3d_box_batch_lwh_tensor(pred_locs[..., 3:], 
                            torch.zeros_like(pred_locs[..., 0]), 
                                pred_locs[..., :3])
        
        gt_corners = get_3d_box_batch_lwh_tensor(gt_locs[..., 3:], 
                            torch.zeros_like(gt_locs[..., 0]), 
                                gt_locs[..., :3])
        
        ious = box3d_iou_batch_tensor(
                pred_corners.unsqueeze(1).repeat(1, num_gts, 1, 1).reshape(-1, 8, 3),
                gt_corners.unsqueeze(0).repeat(num_preds, 1, 1, 1).reshape(-1, 8, 3)
                ).reshape(num_preds, num_gts)

        # 将 IOU 阈值条件应用到所有 IOU 值
        valid_mask = (ious.amax(1) > 0.25)
        
        # 使用 valid_mask 和 prefix_len 来创建索引范围
        # valid_indices = (torch.arange(1,3) + prefix_len).view(1, 3) + torch.where(valid_mask)[0].view(-1, 1) * 3
        
        # # 将 valid_index 中指定的范围设为 1
        # valid_index = torch.zeros(total_len)
        # valid_index.index_fill_(0, valid_indices.flatten(), 1)

        return valid_mask
    

def update_caption(caption, assigned_ids):
    new_ids = {int(assigned_id): i for i, assigned_id in enumerate(assigned_ids)}
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        old_id = int(caption[idx+4:idx+7])
        new_id = int(new_ids[old_id])
        caption = caption[:idx+4] + f"{new_id:03}" + caption[idx+7:]
    return caption


def recover_caption(caption, assigned_ids):
    id_format = "<OBJ\\d{3}>"
    for match in re.finditer(id_format, caption):
        idx = match.start()
        new_id = int(caption[idx+4:idx+7])
        try:
            old_id = int(assigned_ids[new_id])
        except:
            old_id = random.randint(0, len(assigned_ids)-1)
        caption = caption[:idx+4] + f"{old_id:03}" + caption[idx+7:]
    return caption


if __name__ == "__main__":
    caption = "<OBJ001> <OBJ002>"
    assigned_ids = torch.randperm(5)
    print(assigned_ids)
    caption = update_caption(caption, assigned_ids)
    print(caption)
    caption = recover_caption(caption, assigned_ids)
    print(caption)