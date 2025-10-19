import torch
import random
import sys
sys.path.append('.')
from utils.box_utils import box3d_iou, construct_bbox_corners
import re
from collections import defaultdict, OrderedDict
import json
import os
import numpy as np
from scipy.optimize import linear_sum_assignment

from utils.helper import clean_answer, answer_match, scanrefer_get_unique_multiple_lookup

scannet_attribute_file = '/data/kangweitai/3D/chat3d-anno/scannet/scannet_val_attributes.pt'
# scannet_attribute_file = '/home/weitai/data/chat3d-scannet/scannet_val_attributes.pt'
scan2cap_gt_cap = '/data/kangweitai/3D/annotation/scan2cap/scan2cap_val_corpus.json'
# scan2cap_gt_cap = '/home/weitai/data/chat3dv2-anno/scan2cap_val_corpus.json'


def split_OBJ(answer):
    # 使用正则表达式查找第一个 <OBJxxx> 的位置
    match = re.search(r'<OBJ\d{3}>', answer)
    
    if match:
        # 找到第一个 <OBJxxx> 的起始位置
        obj_start = match.start()
        
        # 截取从字符串开头到第一个 <OBJxxx> 位置为止的部分
        before_obj = answer[:obj_start]
        after_obj = answer[obj_start:]
        
        # 找到before_obj中最后一个句子结束符的位置（句号、感叹号、问号等）
        last_punctuation_idx = max(before_obj.rfind('.'), before_obj.rfind(','), 
                                   before_obj.rfind(';'), before_obj.rfind('?'), before_obj.rfind('!'))
        
        if last_punctuation_idx != -1:
            # Caption部分为最后一个标点符号之前的内容
            first_part = before_obj[:last_punctuation_idx+1].strip()
            # 其余部分（包括第一个 <OBJxxx> 及其之后）为 second_part
            second_part = (before_obj[last_punctuation_idx+1:] + after_obj).strip()
        else:
            # 如果在before_obj中没有找到标点符号，那么全部视为caption
            first_part = before_obj.strip()
            second_part = after_obj.strip()
    else:
        # 如果没有找到 <OBJxxx>，那么整个answer就是caption
        first_part = answer.strip()
        second_part = ''
    
    # 确保 caption 部分以句号结尾
    if not first_part.endswith('.'):
        first_part = re.sub(r'[^\w\s]$', '.', first_part) if re.search(r'[^\w\s]$', first_part) else first_part + '.'

    return [first_part, second_part]


def calc_scanrefer_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['scanrefer'][4]
    # scannet_attribute_file = "annotations/scannet_val_attributes.pt"

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    unique_multiple_lookup = scanrefer_get_unique_multiple_lookup()

    iou25_acc = 0
    iou50_acc = 0
    unique_iou25_acc = 0
    unique_iou50_acc = 0
    unique_all = 0
    multiple_iou25_acc = 0
    multiple_iou50_acc = 0
    multiple_all = 0

    # count_list = [0] * 150
    # iou25_acc_list = [0] * 150
    # iou50_acc_list = [0] * 150
    id_format = "<OBJ\\d{3}>"

    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        unique_multiple = unique_multiple_lookup[scene_id][str(obj_id)]
        if unique_multiple == 0:
            unique_all += 1
        else:
            multiple_all += 1
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
        pred_id = 0
        for match in re.finditer(id_format, pred):
            idx = match.start()
            cur_id = int(pred[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        pred_locs = instance_locs[pred_id].tolist()
        gt_locs = scannet_locs[obj_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        if iou >= 0.25:
            iou25_acc += 1
            if unique_multiple == 0:
                unique_iou25_acc += 1
            else:
                multiple_iou25_acc += 1
            # iou25_acc_list[scannet_locs.shape[0]] += 1
        if iou >= 0.5:
            iou50_acc += 1
            if unique_multiple == 0:
                unique_iou50_acc += 1
            else:
                multiple_iou50_acc += 1
            # iou50_acc_list[scannet_locs.shape[0]] += 1
        # count_list[scannet_locs.shape[0]] += 1

    val_scores = {
        '[scanrefer] Acc@0.25': float(iou25_acc) / len(preds),
        '[scanrefer] Acc@0.50': float(iou50_acc) / len(preds),
        # '[scanrefer] Unique Acc@0.25': float(unique_iou25_acc) / unique_all,
        # '[scanrefer] Unique Acc@0.50': float(unique_iou50_acc) / unique_all,
        # '[scanrefer] Multiple Acc@0.25': float(multiple_iou25_acc) / multiple_all,
        # '[scanrefer] Multiple Acc@0.50': float(multiple_iou50_acc) / multiple_all
    }

    return val_scores


def calc_multi3dref_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['multi3dref'][4]

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    id_format = "<OBJ\\d{3}>"

    evaluation_types = {"zt_w_d": 0, "zt_wo_d": 1, "st_w_d": 2, "st_wo_d": 3, "mt": 4}
    eval_type_mask = np.empty(len(preds), dtype=np.uint8)
    # iou_25_f1_scores = np.empty(len(preds), dtype=np.float32)
    # iou_50_f1_scores = np.empty(len(preds), dtype=np.float32)
    iou_25_f1_scores = defaultdict(list)
    iou_50_f1_scores = defaultdict(list)

    for i, pred in enumerate(preds):
        scene_id = pred['scene_id']
        obj_id = pred['gt_id']
        gt_ids = pred['ref_captions']
        pred_sentence = pred['pred']
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]
        pred_ids = []
        for match in re.finditer(id_format, pred_sentence):
            idx = match.start()
            cur_id = int(pred_sentence[idx+4:idx+7])
            if cur_id < instance_num and cur_id < 150:
                pred_ids.append(cur_id)
        eval_type = pred['type_info']
        eval_type_mask[i] = evaluation_types[eval_type]
        iou_25_f1, iou_50_f1 = 0, 0
        if eval_type in ['zt_wo_d', 'zt_w_d']:
            if len(pred_ids) == 0:
                iou_25_f1 = iou_50_f1 = 1
            else:
                iou_25_f1 = iou_50_f1 = 0
        else:
            pred_corners_list = []
            gt_corners_list = []
            for pred_id in pred_ids:
                pred_locs = instance_locs[pred_id].tolist()
                pred_corners_list.append(construct_bbox_corners(pred_locs[:3], pred_locs[3:]))
            for gt_id in gt_ids:
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners_list.append(construct_bbox_corners(gt_locs[:3], gt_locs[3:]))
            square_matrix_len = max(len(pred_ids), len(gt_ids))
            iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
            for pred_idx, pred_corners in enumerate(pred_corners_list):
                for gt_idx, gt_corners in enumerate(gt_corners_list):
                    iou_matrix[pred_idx, gt_idx] = box3d_iou(pred_corners, gt_corners)
            iou_25_tp = 0
            iou_50_tp = 0
            row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)
            for ii in range(len(pred_ids)):
                iou = iou_matrix[row_idx[ii], col_idx[ii]]
                if iou >= 0.25:
                    iou_25_tp += 1
                if iou >= 0.5:
                    iou_50_tp += 1
            iou_25_f1 = 2 * iou_25_tp / (len(pred_ids) + len(gt_ids))
            iou_50_f1 = 2 * iou_50_tp / (len(pred_ids) + len(gt_ids))
        iou_25_f1_scores['all'].append(iou_25_f1)
        iou_50_f1_scores['all'].append(iou_50_f1)
        iou_25_f1_scores[eval_type].append(iou_25_f1)
        iou_50_f1_scores[eval_type].append(iou_50_f1)

    val_scores = {}
    for k in iou_25_f1_scores.keys():
        val_scores[f"[multi3dref] {k} F1@0.25"] = np.mean(iou_25_f1_scores[k])
        val_scores[f"[multi3dref] {k} F1@0.50"] = np.mean(iou_50_f1_scores[k])
    return val_scores


def calc_scan2cap_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['scan2cap'][4]

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    gt_dict = json.load(open(scan2cap_gt_cap))
    tmp_preds_iou25 = {}
    tmp_preds_iou50 = {}
    tmp_targets = {}
    for pred in preds:
        scene_id = pred['scene_id']
        pred_id = pred['pred_id']
        gt_id = pred['gt_id']
        pred_locs = instance_attrs[scene_id]['locs'][pred_id].tolist()
        gt_locs = scannet_attrs[scene_id]['locs'][gt_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        key = f"{scene_id}|{gt_id}"
        if iou >= 0.25:
            tmp_preds_iou25[key] = [{'caption': f"sos {pred['pred']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou25[key] = [{'caption': f"sos eos"}]
        if iou >= 0.5:
            tmp_preds_iou50[key] = [{'caption': f"sos {pred['pred']} eos".replace('\n', ' ')}]
        else:
            tmp_preds_iou50[key] = [{'caption': f"sos eos"}]
        tmp_targets[key] = [{'caption': caption} for caption in gt_dict[key]]
    
    missing_keys = gt_dict.keys() - tmp_targets.keys()

    for missing_key in missing_keys:
        tmp_preds_iou25[missing_key] = [{'caption': "sos eos"}]
        tmp_preds_iou50[missing_key] = [{'caption': "sos eos"}]
        tmp_targets[missing_key] = [{'caption': caption} for caption in gt_dict[missing_key]]
    
    tmp_preds_iou25 = tokenizer.tokenize(tmp_preds_iou25)
    tmp_preds_iou50 = tokenizer.tokenize(tmp_preds_iou50)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    val_scores = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou25)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if m == 'Bleu_4':
                    val_scores[f"[scan2cap] {m}@0.25"] = sc
        else:
            if method == 'CIDEr':
                val_scores[f"[scan2cap] {method}@0.25"] = score
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds_iou50)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if m == 'Bleu_4':
                    val_scores[f"[scan2cap] {m}@0.50"] = sc
        else:
            if method == 'CIDEr':
                val_scores[f"[scan2cap] {method}@0.50"] = score
    return val_scores


def calc_scanqa_score(preds, tokenizer, scorers, config=None):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    acc, refined_acc = 0, 0
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        item_id = f"{output['scene_id']}_{output['gt_id']}_{output['qid']}_{i}"
        pred = output["pred"]
        pred, pred_obj = split_OBJ(pred)
        if len(pred) > 1:
            if pred[-1] == '.':
                pred = pred[:-1]
            pred = pred[0].lower() + pred[1:]
        # pred = clean_answer(pred)
        # ref_captions = [clean_answer(caption) for caption in output['ref_captions']]
        ref_captions = [caption for caption in output['ref_captions']]
        tmp_acc, tmp_refined_acc = answer_match(pred, ref_captions)
        acc += tmp_acc
        refined_acc += tmp_refined_acc
        tmp_preds[item_id] = [{'caption': pred}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    # acc = acc / len(preds)
    # refined_acc = refined_acc / len(preds)
    # val_scores["[scanqa] EM1"] = acc
    # val_scores["[scanqa] EM1_refined"] = refined_acc
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tmp_targets, tmp_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                # if m == 'Bleu_4':
                val_scores[f"[scanqa] {m}"] = sc
        else:
        #     if method == 'CIDEr':
        #         val_scores[f"[scanqa] {method}"] = score
            val_scores[f"[scanqa] {method}"] = score
    return val_scores


def calc_sqa3d_score(preds, tokenizer, scorers, config=None, eval_name='sqa3d'):
    val_scores = {}
    tmp_preds = {}
    tmp_targets = {}
    metrics = {
        'type0_count': 1e-10, 'type1_count': 1e-10, 'type2_count': 1e-10,
        'type3_count': 1e-10, 'type4_count': 1e-10, 'type5_count': 1e-10,
    }
    em_overall = 0
    em_refined_overall = 0
    em_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    em_refined_type = {0: 0, 1: 0, 2: 0, 3: 0, 4: 0, 5: 0}
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        item_id = f"{output['scene_id']}_{output['gt_id']}_{output['qid']}_{i}"
        pred = output["pred"]
        if len(pred) > 1:
            if pred[-1] == '.':
                pred = pred[:-1]
            pred = pred[0].lower() + pred[1:]
        pred = clean_answer(pred)
        ref_captions = [clean_answer(caption) for caption in output['ref_captions']]
        em_flag, em_refined_flag = answer_match(pred, ref_captions)
        em_overall += em_flag
        em_refined_overall += em_refined_flag
        sqa_type = int(output['type_info'])
        em_type[sqa_type] += em_flag
        em_refined_type[sqa_type] += em_refined_flag
        metrics[f'type{sqa_type}_count'] += 1
        tmp_preds[item_id] = [{'caption': pred}]
        ref_captions = [p.replace("\n", " ").strip() for p in ref_captions]
        tmp_targets[item_id] = [{'caption': caption} for caption in ref_captions]
    tmp_preds = tokenizer.tokenize(tmp_preds)
    tmp_targets = tokenizer.tokenize(tmp_targets)
    em_overall = em_overall / len(preds)
    em_refined_overall = em_refined_overall / len(preds)
    val_scores[f"[{eval_name}] EM1"] = em_overall
    val_scores[f"[{eval_name}] EM1_refined"] = em_refined_overall
    # for key in em_type.keys():
    #     val_scores[f'[sqa3d] EM_type{key}'] = em_type[key] / metrics[f'type{key}_count']
    #     val_scores[f'[sqa3d] EM_refined_type{key}'] = em_refined_type[key] / metrics[f'type{key}_count']
    # for scorer, method in scorers:
    #     score, scores = scorer.compute_score(tmp_targets, tmp_preds)
    #     if type(method) == list:
    #         for sc, scs, m in zip(score, scores, method):
    #             val_scores[f"[sqa3d] {m}"] = sc
    #     else:
    #         val_scores[f"[sqa3d] {method}"] = score
    return val_scores

def check_length(pred, tgt):
    # in case the length of the sentence is less than 4, which will affect 4-gram
    for key, value_list in tgt.items():
        if any(len(value.split(' ')) < 4 for value in value_list):
            num2add = 4 - min([len(value.split(' ')) for value in value_list])
            tgt[key] = [value + num2add * ' eos' for value in tgt[key]]
            pred[key] = [value + num2add * ' eos' for value in pred[key]]

def calculate_score(preds, tgt, name, val_scores, scorers, tokenizer, iou):
    check_miss(preds, tgt)
    tokenized_preds = tokenizer.tokenize(preds)
    tokenized_tgt = tokenizer.tokenize(tgt)
    # if any(x in name for x in ['pointedcap']):
    #     check_length(tokenized_preds, tokenized_tgt)
    # check_length(tokenized_preds, tokenized_tgt)
    if any(x in name for x in ['pointedcap']):
        for scorer, method in scorers:
            score, scores = scorer.compute_score(tokenized_tgt, tokenized_preds)
            if type(method) == list:
                for sc, scs, m in zip(score, scores, method):
                    if m == 'Bleu_4':
                        val_scores[f"{[name]} {m}@{iou}"] = int(sc * 10000) / 10000
            else:
                if method == 'CIDEr':
                    val_scores[f"{[name]} {method}@{iou}"] = int(sc * 10000) / 10000

def calculate_score_list(preds, tgt, scorers, tokenizer):
    check_miss(preds, tgt)
    tokenized_preds = tokenizer.tokenize(preds)
    tokenized_tgt = tokenizer.tokenize(tgt)
    # check_length(tokenized_preds, tokenized_tgt)
    return_list = {}
    for scorer, method in scorers:
        score, scores = scorer.compute_score(tokenized_tgt, tokenized_preds)
        if type(method) == list:
            for sc, scs, m in zip(score, scores, method):
                if m == 'Bleu_1':
                    return_list['Bleu_1'] = np.array(scs)
    
    # EM score list
    em_overall = []
    em_refined_overall = []
    for key in tokenized_tgt.keys():
        em_flag, em_refined_flag = answer_match(tokenized_preds[key][0], tokenized_tgt[key])
        em_overall.append(em_flag)
        em_refined_overall.append(em_refined_flag)

    return_list["EM"] = np.array(em_overall)
    return_list["EM_refined"] = np.array(em_refined_overall)

    return return_list

def check_miss(preds, tgt):
    missing_keys = tgt.keys() - preds.keys()
    for missing_key in missing_keys:
        preds[missing_key] = [{'caption': ""}]

def calc_partialref_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['partialref'][4]
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    val_scores = {}
    tmp_targets_all = {}
    tmp_targets_zero = {}
    tmp_targets_partial = {}
    tmp_targets_single = {}
    tmp_preds_iou25_all, tmp_preds_iou25_zero, tmp_preds_iou25_partial, tmp_preds_iou25_single = {}, {}, {}, {}
    tmp_preds_iou50_all, tmp_preds_iou50_zero, tmp_preds_iou50_partial, tmp_preds_iou50_single = {}, {}, {}, {}
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        key = f"{i}"
        pred = output["pred"]
        pred_cap, pred_obj = split_OBJ(pred)
        scene_id = output["scene_id"]
        obj_id = int(output["gt_id"])
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]

        ref_captions = [f"{split_OBJ(caption)[0]}" for caption in output['ref_captions']]
        tmp_targets_all[key] = [{'caption': caption} for caption in ref_captions]
        if 'no' in ref_captions[0].lower():
            tmp_targets_zero[key] = [{'caption': caption} for caption in ref_captions]
        elif 'yes' in ref_captions[0].lower():
            tmp_targets_single[key] = [{'caption': caption} for caption in ref_captions]
        else:
            tmp_targets_partial[key] = [{'caption': caption} for caption in ref_captions]

        pred_id = -1
        for match in re.finditer("<OBJ\\d{3}>", pred_obj):
            idx = match.start()
            cur_id = int(pred_obj[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        if 'no' in ref_captions[0].lower():
            iou = 1 if pred_id == -1 else 0
        else:
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)

        if any(x in pred_cap.lower() for x in ['no', 'yes', 'it is']):
            if iou >= 0.25:
                tmp_preds_iou25_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                if 'no' in pred_cap.lower() and key in tmp_targets_zero:
                    tmp_preds_iou25_zero[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                elif 'yes' in pred_cap.lower() and key in tmp_targets_single:
                    tmp_preds_iou25_single[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                elif key in tmp_targets_partial:
                    tmp_preds_iou25_partial[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                
            if iou >= 0.5:
                tmp_preds_iou50_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                if 'no' in pred_cap.lower() and key in tmp_targets_zero:
                    tmp_preds_iou50_zero[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                elif 'yes' in pred_cap.lower() and key in tmp_targets_single:
                    tmp_preds_iou50_single[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
                elif key in tmp_targets_partial:
                    tmp_preds_iou50_partial[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
        else:
            # will add missing keys in the calculate_score function
            pass
    
    # calculate all
    calculate_score(tmp_preds_iou25_all, tmp_targets_all, "partialref_all", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_all, tmp_targets_all, "partialref_all", val_scores, scorers, tokenizer, "0.50")
    # calculate zero
    calculate_score(tmp_preds_iou25_zero, tmp_targets_zero, "partialref_zero", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_zero, tmp_targets_zero, "partialref_zero", val_scores, scorers, tokenizer, "0.50")
    # calculate single
    calculate_score(tmp_preds_iou25_single, tmp_targets_single, "partialref_single", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_single, tmp_targets_single, "partialref_single", val_scores, scorers, tokenizer, "0.50")
    # calculate partial
    calculate_score(tmp_preds_iou25_partial, tmp_targets_partial, "partialref_partial", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_partial, tmp_targets_partial, "partialref_partial", val_scores, scorers, tokenizer, "0.50")
    
    # sort val_scores base on the key
    val_scores = OrderedDict(sorted(val_scores.items(), key=lambda x: x[0]))
    return val_scores

def calc_pointedcap_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['pointedcap'][4]
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    val_scores = {}
    tmp_targets_all = {}
    tmp_targets_pos = {}
    tmp_targets_neg = {}
    tmp_preds_iou25_all, tmp_preds_iou25_pos, tmp_preds_iou25_neg = {}, {}, {}
    tmp_preds_iou50_all, tmp_preds_iou50_pos, tmp_preds_iou50_neg = {}, {}, {}
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        key = f"{i}"
        pred = output["pred"]
        pred_cap, pred_obj = split_OBJ(pred)
        scene_id = output["scene_id"]
        obj_id = int(output["gt_id"])
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]

        ref_captions = [f"{split_OBJ(caption)[0]}" for caption in output['ref_captions']]
        tmp_targets_all[key] = [{'caption': caption} for caption in ref_captions]
        if 'no' in ref_captions[0].lower():
            tmp_targets_neg[key] = [{'caption': caption} for caption in ref_captions]
        else:
            tmp_targets_pos[key] = [{'caption': caption} for caption in ref_captions]

        pred_id = -1
        for match in re.finditer("<OBJ\\d{3}>", pred_obj):
            idx = match.start()
            cur_id = int(pred_obj[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        if 'no' in ref_captions[0].lower():
            iou = 1 if pred_id == -1 else 0
        else:
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)

        if iou >= 0.25:
            tmp_preds_iou25_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            if 'no' in pred_cap.lower() and key in tmp_targets_neg:
                tmp_preds_iou25_neg[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            elif key in tmp_targets_pos:
                tmp_preds_iou25_pos[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            
        if iou >= 0.5:
            tmp_preds_iou50_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            if 'no' in pred_cap.lower() and key in tmp_targets_neg:
                tmp_preds_iou50_neg[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            elif key in tmp_targets_pos:
                tmp_preds_iou50_pos[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]

    # calculate all
    if len(tmp_targets_all) > 0:
        calculate_score(tmp_preds_iou25_all, tmp_targets_all, "pointedcap_all", val_scores, scorers, tokenizer, "0.25")
        calculate_score(tmp_preds_iou50_all, tmp_targets_all, "pointedcap_all", val_scores, scorers, tokenizer, "0.50")
    # calculate neg
    if len(tmp_preds_iou25_neg) > 0:
        calculate_score(tmp_preds_iou25_neg, tmp_preds_iou25_neg, "pointedcap_neg", val_scores, scorers, tokenizer, "0.25")
        calculate_score(tmp_preds_iou50_neg, tmp_preds_iou25_neg, "pointedcap_neg", val_scores, scorers, tokenizer, "0.50")
    # calculate pos
    if len(tmp_targets_pos) > 0:
        calculate_score(tmp_preds_iou25_pos, tmp_targets_pos, "pointedcap_pos", val_scores, scorers, tokenizer, "0.25")
        calculate_score(tmp_preds_iou50_pos, tmp_targets_pos, "pointedcap_pos", val_scores, scorers, tokenizer, "0.50")
    
    # sort val_scores base on the key
    val_scores = OrderedDict(sorted(val_scores.items(), key=lambda x: x[0]))
    return val_scores

def calc_taskqa_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['taskqa'][4]
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    val_scores = {}
    tmp_targets_all = {}
    tmp_targets_pos = {}
    tmp_targets_neg = {}
    tmp_preds_iou25_all, tmp_preds_iou25_pos, tmp_preds_iou25_neg = {}, {}, {}
    tmp_preds_iou50_all, tmp_preds_iou50_pos, tmp_preds_iou50_neg = {}, {}, {}
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        key = f"{i}"
        pred = output["pred"]
        pred_cap, pred_obj = split_OBJ(pred)
        scene_id = output["scene_id"]
        obj_id = int(output["gt_id"])
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]

        ref_captions = [f"{split_OBJ(caption)[0]}" for caption in output['ref_captions']]
        tmp_targets_all[key] = [{'caption': caption} for caption in ref_captions]
        if 'no' in ref_captions[0].lower():
            tmp_targets_neg[key] = [{'caption': caption} for caption in ref_captions]
        else:
            tmp_targets_pos[key] = [{'caption': caption} for caption in ref_captions]

        pred_id = -1
        for match in re.finditer("<OBJ\\d{3}>", pred_obj):
            idx = match.start()
            cur_id = int(pred_obj[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        if 'no' in ref_captions[0].lower():
            iou = 1 if pred_id == -1 else 0
        else:
            pred_locs = instance_locs[pred_id].tolist()
            gt_locs = scannet_locs[obj_id].tolist()
            pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
            gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
            iou = box3d_iou(pred_corners, gt_corners)

        if iou >= 0.25:
            tmp_preds_iou25_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            if 'no' in pred_cap.lower() and key in tmp_targets_neg:
                tmp_preds_iou25_neg[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            elif key in tmp_targets_pos:
                tmp_preds_iou25_pos[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            
        if iou >= 0.5:
            tmp_preds_iou50_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            if 'no' in pred_cap.lower() and key in tmp_targets_neg:
                tmp_preds_iou50_neg[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
            elif key in tmp_targets_pos:
                tmp_preds_iou50_pos[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
        
    # calculate all
    calculate_score(tmp_preds_iou25_all, tmp_targets_all, "taskqa_all", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_all, tmp_targets_all, "taskqa_all", val_scores, scorers, tokenizer, "0.50")
    # calculate neg
    calculate_score(tmp_preds_iou25_neg, tmp_preds_iou25_neg, "taskqa_neg", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_neg, tmp_preds_iou25_neg, "taskqa_neg", val_scores, scorers, tokenizer, "0.50")
    # calculate pos
    calculate_score(tmp_preds_iou25_pos, tmp_targets_pos, "taskqa_pos", val_scores, scorers, tokenizer, "0.25")
    calculate_score(tmp_preds_iou50_pos, tmp_targets_pos, "taskqa_pos", val_scores, scorers, tokenizer, "0.50")

    # sort val_scores base on the key
    val_scores = OrderedDict(sorted(val_scores.items(), key=lambda x: x[0]))
    return val_scores

def calc_groundedqa_score(preds, tokenizer, scorers, config=None):
    instance_attribute_file = config.val_file_dict['groundedqa'][4]
    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')
    val_scores = {}
    tmp_targets_all = OrderedDict()
    tmp_targets_pos = OrderedDict()
    tmp_targets_neg = OrderedDict()
    tmp_preds_all, tmp_preds_pos, tmp_preds_neg = OrderedDict(), OrderedDict(), OrderedDict()
    f1_25_list_all = []
    f1_50_list_all = []
    f1_25_list_pos = []
    f1_50_list_pos = []
    f1_25_list_neg = []
    f1_50_list_neg = []
    print("Total samples:", len(preds))
    for i, output in enumerate(preds):
        key = f"{i}"
        pred = output["pred"]
        pred_cap, pred_obj = split_OBJ(pred)
        pred_cap = clean_answer(pred_cap)
        scene_id = output["scene_id"]
        gt_ids = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        instance_num = instance_locs.shape[0]

        ref_captions = [f"{clean_answer(split_OBJ(caption)[0])}" for caption in output['ref_captions']]
        tmp_targets_all[key] = [{'caption': caption} for caption in ref_captions]
        if 'no' in ref_captions[0].lower():
            tmp_targets_neg[key] = [{'caption': caption} for caption in ref_captions]
        else:
            tmp_targets_pos[key] = [{'caption': caption} for caption in ref_captions]

        pred_ids = []
        for match in re.finditer("<OBJ\\d{3}>", pred_obj):
            idx = match.start()
            cur_id = int(pred_obj[idx+4:idx+7])
            if cur_id < instance_num:
                pred_ids.append(cur_id)
        if 'no' in ref_captions[0].lower():
            f1_25 = 1 if len(pred_ids) == 0 else 0
            f1_50 = f1_25
            f1_25_list_neg.append(f1_25)
            f1_50_list_neg.append(f1_50)
        else:
            pred_corners_list = []
            gt_corners_list = []
            for pred_id in pred_ids:
                pred_locs = instance_locs[pred_id].tolist()
                pred_corners_list.append(construct_bbox_corners(pred_locs[:3], pred_locs[3:]))
            for gt_id in gt_ids:
                gt_locs = scannet_locs[gt_id].tolist()
                gt_corners_list.append(construct_bbox_corners(gt_locs[:3], gt_locs[3:]))
            square_matrix_len = max(len(pred_ids), len(gt_ids))
            iou_matrix = np.zeros(shape=(square_matrix_len, square_matrix_len), dtype=np.float32)
            for pred_idx, pred_corners in enumerate(pred_corners_list):
                for gt_idx, gt_corners in enumerate(gt_corners_list):
                    iou_matrix[pred_idx, gt_idx] = box3d_iou(pred_corners, gt_corners)
            iou_25_tp = 0
            iou_50_tp = 0
            row_idx, col_idx = linear_sum_assignment(iou_matrix * -1)
            for ii in range(len(pred_ids)):
                iou = iou_matrix[row_idx[ii], col_idx[ii]]
                if iou >= 0.25:
                    iou_25_tp += 1
                if iou >= 0.5:
                    iou_50_tp += 1
            f1_25 = 2 * iou_25_tp / (len(pred_ids) + len(gt_ids))
            f1_50 = 2 * iou_50_tp / (len(pred_ids) + len(gt_ids))

            f1_25_list_pos.append(f1_25)
            f1_50_list_pos.append(f1_50)

        f1_25_list_all.append(f1_25)
        f1_50_list_all.append(f1_50)
        
        tmp_preds_all[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
        if 'no' in pred_cap.lower() and key in tmp_targets_neg:
            tmp_preds_neg[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
        elif key in tmp_targets_pos:
            tmp_preds_pos[key] = [{'caption': f"{pred_cap}".replace('\n', ' ')}]
    
    record_name = ['Bleu_1', 'EM', 'EM_refined']
    # calculate all
    res = calculate_score_list(tmp_preds_all, tmp_targets_all, scorers, tokenizer)
    f1_25_list_all = np.array(f1_25_list_all)
    f1_50_list_all = np.array(f1_50_list_all)
    for name in record_name:
        val_scores[f"[groundedqa_all] {name}@0.25"] = np.mean(res[name] * f1_25_list_all)
        val_scores[f"[groundedqa_all] {name}@0.50"] = np.mean(res[name] * f1_50_list_all)
    # calculate pos
    res = calculate_score_list(tmp_preds_pos, tmp_targets_pos, scorers, tokenizer)
    f1_25_list_pos = np.array(f1_25_list_pos)
    f1_50_list_pos = np.array(f1_50_list_pos)
    for name in record_name:
        val_scores[f"[groundedqa_pos] {name}@0.25"] = np.mean(res[name] * f1_25_list_pos)
        val_scores[f"[groundedqa_pos] {name}@0.50"] = np.mean(res[name] * f1_50_list_pos)
    # calculate neg
    res = calculate_score_list(tmp_preds_neg, tmp_targets_neg, scorers, tokenizer)
    f1_25_list_neg = np.array(f1_25_list_neg)
    f1_50_list_neg = np.array(f1_50_list_neg)
    for name in record_name:
        val_scores[f"[groundedqa_neg] {name}@0.25"] = np.mean(res[name] * f1_25_list_neg)
        val_scores[f"[groundedqa_neg] {name}@0.50"] = np.mean(res[name] * f1_50_list_neg)
    
    # sort val_scores base on the key
    val_scores = OrderedDict(sorted(val_scores.items(), key=lambda x: x[0]))
    return val_scores

def calc_nr3d_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['nr3d'][4]

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    iou25_acc = 0
    iou50_acc = 0

    id_format = "<OBJ\\d{3}>"
    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
        pred_id = 0
        for match in re.finditer(id_format, pred):
            idx = match.start()
            cur_id = int(pred[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        pred_locs = instance_locs[pred_id].tolist()
        gt_locs = scannet_locs[obj_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        if iou >= 0.25:
            iou25_acc += 1
        if iou >= 0.5:
            iou50_acc += 1

    val_scores = {
        '[nr3d] Acc@0.25': float(iou25_acc) / len(preds),
        '[nr3d] Acc@0.50': float(iou50_acc) / len(preds),
    }

    return val_scores

def calc_sr3d_score(preds, config=None):
    instance_attribute_file = config.val_file_dict['sr3d'][4]

    instance_attrs = torch.load(instance_attribute_file, map_location='cpu')
    scannet_attrs = torch.load(scannet_attribute_file, map_location='cpu')

    iou25_acc = 0
    iou50_acc = 0

    id_format = "<OBJ\\d{3}>"
    for i, output in enumerate(preds):
        scene_id = output["scene_id"]
        obj_id = output["gt_id"]
        instance_locs = instance_attrs[scene_id]["locs"]
        scannet_locs = scannet_attrs[scene_id]["locs"]
        pred = output["pred"]
        instance_num = instance_locs.shape[0]
        pred_id = 0
        for match in re.finditer(id_format, pred):
            idx = match.start()
            cur_id = int(pred[idx+4:idx+7])
            if cur_id < instance_num:
                pred_id = cur_id
                break
        pred_locs = instance_locs[pred_id].tolist()
        gt_locs = scannet_locs[obj_id].tolist()
        pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
        gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
        iou = box3d_iou(pred_corners, gt_corners)
        if iou >= 0.25:
            iou25_acc += 1
        if iou >= 0.5:
            iou50_acc += 1

    val_scores = {
        '[sr3d] Acc@0.25': float(iou25_acc) / len(preds),
        '[sr3d] Acc@0.50': float(iou50_acc) / len(preds),
    }

    return val_scores

if __name__ == '__main__':
    from pycocoevalcap.bleu.bleu import Bleu
    from pycocoevalcap.meteor.meteor import Meteor
    from pycocoevalcap.rouge.rouge import Rouge
    from pycocoevalcap.cider.cider import Cider
    # from pycocoevalcap.spice.spice import Spice
    from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
    saved_preds = json.load(open('/mnt/petrelfs/huanghaifeng/share/Chat-3D-v2/outputs/20240504_213518_lr5e-6_ep3_scanrefer#scan2cap#obj_align#scanqa#sqa3d#multi3dref#scannet_caption#scannet_region_caption#nr3d_caption__scanrefer#scan2cap#scanqa#multi3dref#sqa3d__mask3d_video/preds_epoch1_step3812_scanqa.json'))
    scorers = [
        (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
        (Meteor(), "METEOR"),
        (Rouge(), "ROUGE_L"),
        (Cider(), "CIDEr"),
        # (Spice(), "SPICE")
    ]
    tokenizer = PTBTokenizer()
    val_scores = calc_scanqa_score(saved_preds, tokenizer=tokenizer, scorers=scorers)
    print(json.dumps(val_scores, indent=4))
