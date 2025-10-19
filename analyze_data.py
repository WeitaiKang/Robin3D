
import json
import torch
import numpy as np
from tqdm import tqdm
from collections import Counter
import matplotlib.pyplot as plt
import spacy
import json
import matplotlib.ticker as ticker
import os
import sys
import numpy as np
import torch
import re

# gpt-4 (gpt-4-0613)
'''
response ChatCompletion(id='chatcmpl-8W4GdQnOfjeIknjlAuyL82GoieCrm', 
choices=[
  Choice(
    finish_reason='stop', 
    index=0, 
    message=
      ChatCompletionMessage
        (content=
        'computer tower = I want to run software applications for work purposes; 
        \n\nradiator = I need to stay warm during colder days while working; 
        \n\nclothing = I want to wear professional attire during office hours; 
        \n\ndesk = I can organize my work files and set up my working equipment;
        \n\nmonitor = I need to display the content of my work; 
        \n\nbackpack = I want to carry my laptop and other work materials; 
        \n\ntrash can = I need to dispose of any waste and keeps the office clean; 
        \n\njacket = I want to put it on when it gets colder in office; 
        \n\nshoes = I want to use them to walk comfortably around the office; 
        \n\ntelephone = I need to make or receive work-related calls; 
        \n\nbook = I want to reference information during my work; 
        \n\nkeyboard = I want to input data in my computer; 
        \n\ncup = I can use it to drink coffee or water during office hours; 
        \n\nwhiteboard = I need to write down important notes during meetings.', 
        role='assistant', function_call=None, tool_calls=None))
        ], 
        created=1702653591, model='gpt-4-0613', object='chat.completion', system_fingerprint=None, 
        usage=CompletionUsage(completion_tokens=202, prompt_tokens=156, total_tokens=358))
ans_dict {
  'computer tower': 'I want to run software applications for work purposes', 
  'radiator': 'I need to stay warm during colder days while working', 
  'clothing': 'I want to wear professional attire during office hours', 
  'desk': 'I can organize my work files and set up my working equipment', 
  'monitor': 'I need to display the content of my work', 
  'backpack': 'I want to carry my laptop and other work materials', 
  'trash can': 'I need to dispose of any waste and keeps the office clean', 
  'jacket': 'I want to put it on when it gets colder in office', 
  'shoes': 'I want to use them to walk comfortably around the office', 
  'telephone': 'I need to make or receive work-related calls', 
  'book': 'I want to reference information during my work', 
  'keyboard': 'I want to input data in my computer', 
  'cup': 'I can use it to drink coffee or water during office hours', 
  'whiteboard': 'I need to write down important notes during meetings.'}

another attempt
{'computer tower': 'I need to process work documents', 
'radiator': 'I want to warm up the cold office room', 
'clothing': 'I need to wear to follow the office dress code', 
'desk': 'I want to place my work tools for better productivity', 
'monitor': 'I need to display information for my work', 
'backpack': 'I want to store and transport my work-related items', 
'trash can': 'I need to discard unwanted items to keep the office clean', 
'jacket': 'I need to wear when feeling cold', 
'shoes': "I want to wear to adhere to the office's dress code", 
'telephone': 'I want to communicate with coworkers or clients', 
'book': 'I desire to gain knowledge during breaks', 
'keyboard': 'I want to input data into the computer', 
'cup': 'I want to drink coffee while working', 
'whiteboard': 'I want to present ideas during a meeting.'}
'''

# gpt-4-1106-preview
'''
response 
ChatCompletion(id='chatcmpl-8W4OPHgi0LVspI7BO7fMUTU9aSC8n', 
choices=[
  Choice(finish_reason='stop', index=0, 
  message=
    ChatCompletionMessage(content=
    '- computer tower = Connect it to the monitor to complete a workstation for various computing tasks.
    \n- radiator = You can use it to warm the room for comfortable working conditions.
    \n- clothing = Organize it in the backpack or hang the jacket for maintaining a neat space and personal attire.
    \n- desk = Arrange the computer tower, monitor, keyboard, books, and cup on it to create an efficient work setup.
    \n- monitor = Pair it with the computer tower for visual display of work and information.
    \n- backpack = Pack it with personal items, including clothing and books, for easy transport.
    \n- trash can = Dispose of litter, such as used cups or unnecessary papers to keep the office tidy.
    \n- jacket = Wear it when leaving the office or if it gets chilly from the radiator.
    \n- shoes = Change into them before leaving the office or for comfort during the workday.
    \n- telephone = Use it to make or receive work-related calls.
    \n- book = Read it for research, personal development, or leisure during breaks.
    \n- keyboard = Type on it to interact with the computer for work tasks.
    \n- cup = Drink from it to stay hydrated while working.
    \n- whiteboard = Write or draw diagrams on it for presentations or to organize thoughts and tasks.', 
    role='assistant', function_call=None, tool_calls=None))], created=1702654073, 
    model='gpt-4-1106-preview', object='chat.completion', system_fingerprint='fp_6aca3b5ce1', 
    usage=CompletionUsage(completion_tokens=248, prompt_tokens=156, total_tokens=404))
'''

# gpt-3.5-turbo-1106
'''
response 
ChatCompletion(id='chatcmpl-8W4XLsfZxtyNGpDShZawr8dVYj3rw', 
choices=[Choice(finish_reason='stop', index=0, 
message=
  ChatCompletionMessage(
    content=
    'computer tower = I need to perform tasks on the computer; 
    radiator = I need to regulate the temperature in the office; 
    clothing = I need to change my outfit for comfort; 
    desk = I should work on tasks and organize my work materials; 
    monitor = I need to display information for work purposes; 
    backpack = I need to carry work-related items; 
    trash can = I need to dispose of unnecessary items; 
    jacket = I need to keep warm during breaks; 
    shoes = I need to wear comfortable footwear for work; 
    telephone = I need to communicate with colleagues and clients; 
    book = I need to read to enhance knowledge; 
    keyboard = I need to input data into the computer; 
    cup = I need to drink beverages while working; 
    whiteboard = I need to visualize ideas and concepts for work.', 
    role='assistant', function_call=None, tool_calls=None))], created=1702654627, 
    model='gpt-3.5-turbo-1106', object='chat.completion', system_fingerprint='fp_f3efa6edfc', 
    usage=CompletionUsage(completion_tokens=156, prompt_tokens=156, total_tokens=312))
ans_dict 
{'computer tower': 'I need to perform tasks on the computer', 
'radiator': 'I need to regulate the temperature in the office', 
'clothing': 'I need to change my outfit for comfort', 
'desk': 'I should work on tasks and organize my work materials', 
'monitor': 'I need to display information for work purposes', 
'backpack': 'I need to carry work-related items', 
'trash can': 'I need to dispose of unnecessary items', 
'jacket': 'I need to keep warm during breaks', 
'shoes': 'I need to wear comfortable footwear for work', 
'telephone': 'I need to communicate with colleagues and clients', 
'book': 'I need to read to enhance knowledge', 
'keyboard': 'I need to input data into the computer', 
'cup': 'I need to drink beverages while working', 
'whiteboard': 'I need to visualize ideas and concepts for work.'}

another attempt
ans_dict 
{
'computer tower': 'I want to store important files and documents on it', 
'radiator': 'I want to keep the office warm and comfortable', 
'clothing': 'I want to wear it to stay comfortable during work hours', 
'desk': 'I want to keep it organized and tidy for productive work', 
'monitor': 'I want to display work and important information on it', 
'backpack': 'I want to carry important items for work', 
'trash can': 'I want to keep the office clean and organized by disposing of waste', 
'jacket': 'I want to stay warm and comfortable during work hours', 
'shoes': 'I want to wear them for comfort and to move around the office', 
'telephone': 'I want to make and receive important work calls', 
'book': 'I want to read and gain knowledge during breaks', 
'keyboard': 'I want to use it for typing and work-related tasks', 
'cup': 'I want to drink beverages to stay hydrated during work', 
'whiteboard': 'I want to use it for brainstorming and visualizing ideas during meetings.'}

another attempt: temperature=1.2
ans_dict {
  'computer tower': 'I want to access digital documents for work', 
  'radiator': 'I want to regulate the room temperature', 
  'clothing': 'I want to change into something more comfortable after a long day at work', 
  'desk': 'I want to work and organize my tasks', 
  'monitor': 'I want to visualize data for my work', 
  'backpack': 'I want to carry my work essentials to and from the office', 
  'trash can': 'I want to dispose of any unnecessary clutter on my desk', 
  'jacket': 'I want to keep warm in a draughty office', 
  'shoes': 'I want to protect my feet and maintain professionalism', 
  'telephone': 'I want to communicate with colleagues and clients', 
  'book': 'I want to read during breaks to relax my mind', 
  'keyboard': 'I want to type up documents and emails', 
  'cup': 'I want to stay hydrated while working', 
  'whiteboard': 'I want to visually brainstorm and plan out ideas.'}
'''
# ----------------------------------- data analysis ----------------------------------- #


# # 读取 JSON 文件
# file_path = '/data/kangweitai/3D/gpt4_t1.2_allsample_all6attempt.json'
# with open(file_path, 'r') as file:
#     data = json.load(file)

# # 统计 scene_type 和 object_type
# scene_type_counter = Counter()
# object_type_counter = Counter()

# for scene_id, (scene_type, objects) in data.items():
#     scene_type_counter[scene_type] += 1
#     for object_type in objects.keys():
#         object_type_counter[object_type] += len(objects[object_type])

# # 按数量降序排序
# sorted_scene_types = dict(sorted(scene_type_counter.items(), key=lambda item: item[1], reverse=True))
# sorted_object_types = dict(sorted(object_type_counter.items(), key=lambda item: item[1], reverse=True))
# print(f'-- len(sorted_scene_types) = {len(sorted_scene_types)}')
# print(f'-- len(sorted_object_types) = {len(sorted_object_types)}')

# # 绘制 scene_type 的直方图
# plt.figure(figsize=(10, 6))
# plt.bar(sorted_scene_types.keys(), sorted_scene_types.values())
# plt.title('Scene Type Histogram')
# plt.xlabel('Scene Type')
# plt.ylabel('Count')
# plt.xticks(rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig('/data/kangweitai/3D/scene_type_histogram.png')

# # 绘制 object 的直方图
# plt.figure(figsize=(10, 6))
# plt.bar(sorted_object_types.keys(), sorted_object_types.values())
# plt.title('Object Type Histogram')
# plt.xlabel('Object Type')
# plt.ylabel('Count')

# # 标记间隔的横坐标标签
# tick_interval = 10  # 您可以根据需要调整这个值
# plt.xticks([i for i in range(len(sorted_object_types.keys())) if i % tick_interval == 0],
#            [key for i, key in enumerate(sorted_object_types.keys()) if i % tick_interval == 0],
#            rotation=45, ha="right")
# plt.tight_layout()
# plt.savefig('/data/kangweitai/3D/object_type_histogram.png')

# ----------------------------------- data analysis ----------------------------------- #
# with open("/data/kangweitai/3D/gpt4_t1.2_allsample_all6attempt.json", 'r') as file:
#     data = json.load(file)

#     object_scene_sentence = 0
#     for scene_id, (scene_type, objects) in data.items():
#         for obj in objects.keys():
#             object_scene_sentence += len(objects[obj])
#     print(f'-- object_scene_sentence = {object_scene_sentence}') # 45156

# --------------- extract seg ---------------- #

# def get_box3d_min_max(corner):
#     ''' Compute min and max coordinates for 3D bounding box
#         Note: only for axis-aligned bounding boxes

#     Input:
#         corners: numpy array (8,3), assume up direction is Z (batch of N samples)
#     Output:
#         box_min_max: an array for min and max coordinates of 3D bounding box IoU

#     '''

#     min_coord = corner.min(axis=0)
#     max_coord = corner.max(axis=0)
#     x_min, x_max = min_coord[0], max_coord[0]
#     y_min, y_max = min_coord[1], max_coord[1]
#     z_min, z_max = min_coord[2], max_coord[2]

#     return x_min, x_max, y_min, y_max, z_min, z_max


# def box3d_iou(corners1, corners2):
#     ''' Compute 3D bounding box IoU.

#     Input:
#         corners1: numpy array (8,3), assume up direction is Z
#         corners2: numpy array (8,3), assume up direction is Z
#     Output:
#         iou: 3D bounding box IoU

#     '''

#     x_min_1, x_max_1, y_min_1, y_max_1, z_min_1, z_max_1 = get_box3d_min_max(corners1)
#     x_min_2, x_max_2, y_min_2, y_max_2, z_min_2, z_max_2 = get_box3d_min_max(corners2)
#     xA = np.maximum(x_min_1, x_min_2)
#     yA = np.maximum(y_min_1, y_min_2)
#     zA = np.maximum(z_min_1, z_min_2)
#     xB = np.minimum(x_max_1, x_max_2)
#     yB = np.minimum(y_max_1, y_max_2)
#     zB = np.minimum(z_max_1, z_max_2)
#     inter_vol = np.maximum((xB - xA), 0) * np.maximum((yB - yA), 0) * np.maximum((zB - zA), 0)
#     box_vol_1 = (x_max_1 - x_min_1) * (y_max_1 - y_min_1) * (z_max_1 - z_min_1)
#     box_vol_2 = (x_max_2 - x_min_2) * (y_max_2 - y_min_2) * (z_max_2 - z_min_2)
#     iou = inter_vol / (box_vol_1 + box_vol_2 - inter_vol + 1e-8)

#     return iou


# def construct_bbox_corners(center, box_size):
#     sx, sy, sz = box_size
#     x_corners = [sx / 2, sx / 2, -sx / 2, -sx / 2, sx / 2, sx / 2, -sx / 2, -sx / 2]
#     y_corners = [sy / 2, -sy / 2, -sy / 2, sy / 2, sy / 2, -sy / 2, -sy / 2, sy / 2]
#     z_corners = [sz / 2, sz / 2, sz / 2, sz / 2, -sz / 2, -sz / 2, -sz / 2, -sz / 2]
#     corners_3d = np.vstack([x_corners, y_corners, z_corners])
#     corners_3d[0, :] = corners_3d[0, :] + center[0]
#     corners_3d[1, :] = corners_3d[1, :] + center[1]
#     corners_3d[2, :] = corners_3d[2, :] + center[2]
#     corners_3d = np.transpose(corners_3d)

#     return corners_3d

# def default(o):
#     if isinstance(o, np.int64):
#         return int(o)
#     raise TypeError

# --------------- ---------------- #

# scannet_pointgroup_attribute_train = torch.load("/data/kangweitai/3D/chat-3d-v2/annotations/scannet_pointgroup_train_attributes.pt")
# scannet_pointgroup_attribute_val = torch.load("/data/kangweitai/3D/chat-3d-v2/annotations/scannet_pointgroup_val_attributes.pt")
# # 检查是否有重复的key
# for key in scannet_pointgroup_attribute_train.keys():
#     if key in scannet_pointgroup_attribute_val.keys():
#         print(f"key {key} is repeated.")
#         raise 123
# scannet_pointgroup_attribute = dict()
# scannet_pointgroup_attribute.update(scannet_pointgroup_attribute_train)
# scannet_pointgroup_attribute.update(scannet_pointgroup_attribute_val)

# scannet_attribute_train = torch.load("/data/kangweitai/3D/chat-3d-v2/annotations/scannet_train_attributes.pt")
# scannet_attribute_val = torch.load("/data/kangweitai/3D/chat-3d-v2/annotations/scannet_val_attributes.pt")
# # 检查是否有重复的key
# for key in scannet_attribute_train.keys():
#     if key in scannet_attribute_val.keys():
#         print(f"key {key} is repeated.")
#         raise 123
# scannet_attribute = dict()
# scannet_attribute.update(scannet_attribute_train)
# scannet_attribute.update(scannet_attribute_val)

# intention_train = json.load(open("/data/kangweitai/3D/intention/train_samples_dict_vg_format_clean_duplicate_sceneType.json", 'r'))
# intention_val = json.load(open("/data/kangweitai/3D/intention/val_samples_dict_vg_format_clean_duplicate_sceneType.json", 'r'))
# intention_test = json.load(open("/data/kangweitai/3D/intention/test_samples_dict_vg_format_clean_duplicate_sceneType.json", 'r'))

# for split, annos_file in enumerate([intention_train, intention_val, intention_test]):
#     new_anno_file = []
#     new_scannet_pointgroup_attribute = dict()
#     new_scannet_attribute = dict()
#     if split != 0:
#         # for val/test, use new_scannet_attribute to get obj_id, no need to change.
#         if split == 1:
#             save_pointgroup_attribute_name = 'scannet_pointgroup_val_attributes_intent_raw.pt'
#             save_attribute_name = 'scannet_val_attributes_intent.pt'
#             save_anno_name = 'chat3dv2_val_intent.json'
#         if split == 2:
#             save_pointgroup_attribute_name = 'scannet_pointgroup_test_attributes_intent_raw.pt'
#             save_attribute_name = 'scannet_test_attributes_intent.pt'
#             save_anno_name = 'chat3dv2_test_intent.json'
#     else:
#         save_pointgroup_attribute_name = 'scannet_pointgroup_train_attributes_intent_raw.pt'
#         save_attribute_name = 'scannet_train_attributes_intent.pt'
#         save_anno_name = 'chat3dv2_train_intent.json'

#     for anno in tqdm(annos_file):
#         new_anno = dict()
#         new_anno['scene_id'] = anno['scene_id']
#         utterance = anno['description']    
#         prompt= f"According to the given description, \"{utterance}\" Please find the most suitable object that match this description. List all its object IDs."
#         new_anno['prompt'] = prompt
#         # "obj_id": 1
#         # "ref_captions": ["Obj07."] # val
#         # "caption": "That includes obj00, obj01, and obj02." # train
#         anno_obj_ids = anno['object_id']
#         anno_scan_id  = anno['scene_id']

#         if split != 0:
#             # for val/test, use new_scannet_attribute to get obj_id, no need to change.
#             new_anno['obj_id'] = anno['object_id']
#             new_anno['ref_captions'] = [f"Obj{str(item).zfill(2)}." for item in anno['object_id']]
#             new_scannet_pointgroup_attribute[anno_scan_id] = scannet_pointgroup_attribute[anno_scan_id]
#             new_scannet_attribute[anno_scan_id] = scannet_attribute[anno_scan_id]
#             new_anno_file.append(new_anno)
#         else:
#             # for train, we need to change the obj_id to pointgroup_id
#             pointgroup_id = []
#             for anno_obj_id in anno_obj_ids:
#                 gt_locs = scannet_attribute[anno_scan_id]['locs'][anno_obj_id].tolist()
#                 all_pointgroup_locs = scannet_pointgroup_attribute[anno_scan_id]['locs'].tolist()
#                 all_ious = []
#                 for pcg_id in range(len(all_pointgroup_locs)):
#                     pred_locs = all_pointgroup_locs[pcg_id]
#                     pred_corners = construct_bbox_corners(pred_locs[:3], pred_locs[3:])
#                     gt_corners = construct_bbox_corners(gt_locs[:3], gt_locs[3:])
#                     iou = box3d_iou(pred_corners, gt_corners)
#                     all_ious.append(iou)

#                 all_ious = np.array(all_ious)
#                 if np.max(all_ious) > 0.5:
#                     if np.argmax(all_ious) in pointgroup_id:
#                         pass
#                     else:
#                         pointgroup_id.append(np.argmax(all_ious))
            
#             if len(pointgroup_id) != 0:
#                 new_anno['obj_id'] = pointgroup_id
#                 if len(pointgroup_id) == 1:
#                     formatted_items = [f'obj{str(item).zfill(2)}' for item in pointgroup_id]
#                     formatted_str = f"That includes {formatted_items[0]}."
#                 else:
#                     formatted_items = [f'obj{str(item).zfill(2)}' for item in pointgroup_id]
#                     formatted_str = f"That includes {', '.join(formatted_items[:-1])}, and {formatted_items[-1]}."
#                 new_anno['caption'] = formatted_str
#                 new_scannet_pointgroup_attribute[anno_scan_id] = scannet_pointgroup_attribute[anno_scan_id]
#                 new_scannet_attribute[anno_scan_id] = scannet_attribute[anno_scan_id]
#                 new_anno_file.append(new_anno)

#     with open(f"/data/kangweitai/3D/chat-3d-v2/annotations/{save_anno_name}", 'w') as file:
#         json.dump(new_anno_file, file, indent=4, default=default)

#     torch.save(new_scannet_pointgroup_attribute, f"/data/kangweitai/3D/chat-3d-v2/annotations/{save_pointgroup_attribute_name}")
#     torch.save(new_scannet_attribute, f"/data/kangweitai/3D/chat-3d-v2/annotations/{save_attribute_name}")


# chat3dv2_train_intent = json.load(open('/data/kangweitai/3D/chat-3d-v2/annotation_intention/chat3dv2_train_intent.json', 'r'))
# chat3dv2_val_intent = json.load(open('/data/kangweitai/3D/chat-3d-v2/annotation_intention/chat3dv2_val_intent.json', 'r'))
# chat3dv2_test_intent = json.load(open('/data/kangweitai/3D/chat-3d-v2/annotation_intention/chat3dv2_test_intent.json', 'r'))

# intent_train_file = json.load(open('/data/kangweitai/3D/intention/train_samples_dict_vg_format_clean_duplicate_sceneType.json', 'r'))
# intent_val_file = json.load(open('/data/kangweitai/3D/intention/val_samples_dict_vg_format_clean_duplicate_sceneType.json', 'r'))
# intent_test_file = json.load(open('/data/kangweitai/3D/intention/test_samples_dict_vg_format_clean_duplicate_sceneType.json', 'r'))
# # system_txt = "A chat between a curious human and an artificial intelligence assistant. The assistant gives succinct answers to the human's questions within a 30-word limit. "
# splits = ['train', 'val', 'test']
# for i, annos_file in enumerate([intent_train_file, intent_val_file, intent_test_file]):
#     new_anno_file = []
#     for anno in tqdm(annos_file):
#         new_anno = anno.copy()
#         utterance = anno['description']    
#         prompt= f"According to the given description, \"{utterance}\", please identify possible objects in an indoor environment that could be used to satisfy the described human intention. Directly answer with the beginning \"They might be: \"."
#         new_anno['vicuna_prompt'] = prompt
#         new_anno_file.append(new_anno)

#     with open(f"/data/kangweitai/3D/intention/{splits[i]}_objName_sceneType_vicuna.json", 'w') as file:
#         json.dump(new_anno_file, file, indent=4, default=default)


# ------------------- count verb & noun ------------------- #
# train = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/train.json', 'r'))
# val = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/val.json', 'r'))
# test = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/test.json', 'r'))


# # 加载英文模型
# nlp = spacy.load("en_core_web_sm")

# # 初始化动词和名词的计数器
# verb_counter = {}
# noun_counter = {}

# # 遍历数据集
# for dataset in [train, val, test]:
#     for anno in tqdm(dataset):
#         # 获取句子
#         text = anno['utterance']
#         # 使用Spacy处理句子
#         doc = nlp(text)
#         # 遍历句子中的每个词
#         for token in doc:
#             # 检查词的词性
#             if token.pos_ == "VERB":
#                 # 动词计数
#                 verb = token.lemma_  # 使用词元形式进行计数
#                 verb_counter[verb] = verb_counter.get(verb, 0) + 1
#             elif token.pos_ == "NOUN":
#                 # 名词计数
#                 noun = token.lemma_  # 使用词元形式进行计数
#                 noun_counter[noun] = noun_counter.get(noun, 0) + 1

# # 打印结果
# print("Verb counts:", verb_counter)
# print("Noun counts:", noun_counter)

# # save the dict
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/verb_counter.json', 'w') as file:
#     json.dump(verb_counter, file, indent=4)
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/noun_counter.json', 'w') as file:
#     json.dump(noun_counter, file, indent=4)


# ------------------- plt the distribution of verb & noun ------------------- #
# import json
# import matplotlib.pyplot as plt

# # 加载动词和名词计数的字典
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/verb_counter.json', 'r') as file:
#     verb_counts = json.load(file)

# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/noun_counter.json', 'r') as file:
#     noun_counts = json.load(file)

# print(f'--- total type of verbs = {len(verb_counts)}')
# print(f'--- total type of nouns = {len(noun_counts)}')

# # 移除特定动词和名词
# for verb in ['want', 'need', 'aim', 'intend']:
#     verb_counts.pop(verb, None)

# for noun in ['I']:
#     noun_counts.pop(noun, None)

# # 将数据按频次降序排序
# sorted_verb_counts = sorted(verb_counts.items(), key=lambda item: item[1], reverse=True)[:100]
# sorted_noun_counts = sorted(noun_counts.items(), key=lambda item: item[1], reverse=True)[:100]

# # 绘制直方图的函数
# def plot_histogram(data, title, tick_spacing=10):
#     plt.figure(figsize=(10, 6))
#     labels, values = zip(*data)
#     indexes = range(len(labels))
#     plt.bar(indexes, values, color='blue', alpha=0.7)
    
#     # # 在大于零的每个柱状图上标上数值
#     # for index, value in enumerate(values):
#     #     if value > 0:
#     #         plt.text(index, value, str(value), ha='center', va='bottom')

#     # plt.xticks(indexes, labels, rotation=45)  # 旋转标签以便于阅读
#     plt.xticks(indexes[::tick_spacing], labels[::tick_spacing], rotation=45, ha="right")
    
#     # 设置x轴刻度间隔
#     ax = plt.gca()
#     ax.xaxis.set_major_locator(ticker.MultipleLocator(tick_spacing))
    
#     plt.ylabel('Frequency')
    
#     # 保存图表为PDF，调整DPI
#     plt.savefig(f'/data/kangweitai/3D/intention/verb_obj_I_intent/{title}_distribution.pdf', format='pdf', dpi=300)
#     plt.tight_layout()  # 调整布局以防止标签重叠
#     plt.close()

# # 分别绘制动词和名词的直方图
# plot_histogram(sorted_verb_counts, 'Verbs')
# plot_histogram(sorted_noun_counts, 'Nouns')

# ------------------- statistic of object with verbs and nouns ------------------- #
# nlp = spacy.load("en_core_web_sm")
# train = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/train.json', 'r'))
# val = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/val.json', 'r'))
# test = json.load(open('/data/kangweitai/3D/intention/verb_obj_I_intent/test.json', 'r'))

# class_dict = {}

# # 遍历数据集
# for dataset in [train, val, test]:
#     for anno in tqdm(dataset):
#         # 获取句子和目标类别
#         text = anno['utterance']
#         target = anno['target']
        
#         # 如果目标类别不在字典中，则初始化
#         if target not in class_dict:
#             class_dict[target] = {'verbs': set(), 'nouns': set()}
        
#         # 使用Spacy处理句子
#         doc = nlp(text)
        
#         # 遍历句子中的每个词
#         for token in doc:
#             # 根据词性分类动词和名词
#             if token.pos_ == 'VERB':
#                 # 添加动词的基本形式到对应类别的列表中
#                 class_dict[target]['verbs'].add(token.lemma_)
#             elif token.pos_ == 'NOUN':
#                 # 添加名词的基本形式到对应类别的列表中
#                 class_dict[target]['nouns'].add(token.lemma_)

# class_dict_serializable = {}
# for target, parts_of_speech in class_dict.items():
#     class_dict_serializable[target] = {
#         'verbs': list(parts_of_speech['verbs']),
#         'nouns': list(parts_of_speech['nouns'])
#     }
    
# # save the dict
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/object_verb_noun.json', 'w') as file:
#     json.dump(class_dict_serializable, file, indent=4)

# ------------------- statistic of frequency of verbs and nouns towards each object ------------------- #
# import json
# import matplotlib.pyplot as plt

# # 加载object_verb_noun.json文件
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/object_verb_noun.json', 'r') as file:
#     class_dict = json.load(file)

# total_num_verbs = 0
# total_num_nouns = 0
# for target, parts_of_speech in class_dict.items():
#     total_num_verbs += len(parts_of_speech['verbs'])
#     total_num_nouns += len(parts_of_speech['nouns'])

# print(f'--- average number of verbs = {total_num_verbs / len(class_dict)}')
# print(f'--- average number of nouns = {total_num_nouns / len(class_dict)}')

# # 准备数据：计算每个目标类别的动词和名词数量
# data = []
# for target, parts_of_speech in class_dict.items():
#     verb_count = len(parts_of_speech['verbs'])
#     noun_count = len(parts_of_speech['nouns'])
#     data.append((target, verb_count, noun_count))

# # 按照动词数量降序排序
# data_sorted_by_verb = sorted(data, key=lambda x: x[1], reverse=True)
# data_sorted_by_noun = sorted(data, key=lambda x: x[2], reverse=True)

# # 分别提取排序后的标签和计数
# labels_verb, verb_counts, _ = zip(*data_sorted_by_verb)
# labels_noun, _, noun_counts = zip(*data_sorted_by_noun)


# def plot_histogram(labels, counts, ylabel, title):
#     plt.figure(figsize=(12, 8))  # 可调整大小以更好地适应标签
#     indexes = range(len(labels))
#     plt.bar(indexes, counts, color='blue', alpha=0.7)
#     plt.ylabel(ylabel, fontweight='bold', fontsize=14)

#     # 设置x轴刻度间隔并加粗标签
#     tick_spacing = 20  # 设置x轴刻度间隔
#     plt.xticks(indexes[::tick_spacing], labels[::tick_spacing], rotation=45, ha="right", fontweight='bold', fontsize=14)
#     plt.yticks(fontweight='bold', fontsize=12)  # 加粗并调整y轴刻度字体大小
    
#     plt.tight_layout()  # 调整布局以防止标签重叠
#     plt.savefig(f'/data/kangweitai/3D/intention/verb_obj_I_intent/{title}.pdf', format='pdf', dpi=300)
#     plt.close()

# # 分别绘制动词和名词的直方图，按数量降序
# plot_histogram(labels_verb, verb_counts, 'Number of Verb', 'VerbNum_Object')
# plot_histogram(labels_noun, noun_counts, 'Number of Noun', 'NounNum_Object')


# -------------------  ------------------- #

# import json
# import pandas as pd

# # 加载object_verb_noun.json文件
# with open('/data/kangweitai/3D/intention/verb_obj_I_intent/object_verb_noun.json', 'r') as file:
#     class_dict = json.load(file)

# total_num_verbs = 0
# total_num_nouns = 0
# for target, parts_of_speech in class_dict.items():
#     total_num_verbs += len(parts_of_speech['verbs'])
#     total_num_nouns += len(parts_of_speech['nouns'])

# # 计算平均数
# average_num_verbs = total_num_verbs / len(class_dict)
# average_num_nouns = total_num_nouns / len(class_dict)

# # 准备数据：计算每个目标类别的动词和名词数量
# data = []
# for target, parts_of_speech in class_dict.items():
#     verb_count = len(parts_of_speech['verbs'])
#     noun_count = len(parts_of_speech['nouns'])
#     data.append((target, verb_count, noun_count))

# data = data[1::10]

# # 创建DataFrame
# df = pd.DataFrame(data, columns=['Target', 'Verb Count', 'Noun Count'])

# # 保存为Excel文件
# df.to_excel('/data/kangweitai/3D/intention/verb_obj_I_intent/target_verb_noun_counts.xlsx', index=False)

# 也可以保存平均数为Excel文件（如果需要）
# df_average = pd.DataFrame({'Average Number of Verbs': [average_num_verbs], 'Average Number of Nouns': [average_num_nouns]})
# df_average.to_excel('/data/kangweitai/3D/intention/verb_obj_I_intent/average_verb_noun_counts.xlsx', index=False)


# ------------------- wordcloud of verbs and nouns ------------------- #
from wordcloud import WordCloud
import matplotlib.pyplot as plt

file_dir ="mask3d150sort-uni3d-dinov2Giant-iou50-oneprompt"
anno_root = f"/data/kangweitai/3D/chat3d-anno/{file_dir}/"

segmentor = "mask3d"
version = ""

train_file_dict = {
    'scanrefer': [
        f"{anno_root}/scanrefer_{segmentor}_train{version}.json",
    ],
    'scan2cap': [
        f"{anno_root}/scan2cap_{segmentor}_train{version}.json",
    ],
    'nr3d_caption': [
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}.json",
    ],
    'obj_align': [
        f"{anno_root}/obj_align_{segmentor}_train{version}.json",
    ],
    'multi3dref': [
        f"{anno_root}/multi3dref_{segmentor}_train{version}.json",
    ],
    'scanqa': [
        f"{anno_root}/scanqa_train.json",
    ],
    'sqa3d': [
        f"{anno_root}/sqa3d_train.json",
    ],
    'scannet_caption': [
        f"{anno_root}/scannet_caption_{segmentor}_train{version}.json",
    ],
    'scannet_region_caption': [
        f"{anno_root}/scannet_region_caption_{segmentor}_train{version}.json",
    ],
    'partialref': [
        f"{anno_root}/partialref_{segmentor}_train{version}.json",
    ],
    'groundedqa': [
        f"{anno_root}/groundedqa_{segmentor}_train{version}.json",
    ],
    'pointedcap': [
        f"{anno_root}/pointedcap_{segmentor}_train{version}.json",
    ],
    'partial_objalign': [
        f"{anno_root}/partial_obj_align_{segmentor}_train{version}.json",
    ],
    'partial_od': [
        f"{anno_root}/partial_od_{segmentor}_train{version}.json",
    ],
    'nr3d': [
        f"{anno_root}/nr3d_{segmentor}_train{version}.json",
    ],
    'sr3d+': [
        f"{anno_root}/sr3d+_{segmentor}_train{version}.json",
    ],
    'partialrefv2': [
        f"{anno_root}/partialref_train_rephrase_merged.json",
    ],
    'pointedcapv2': [
        f"{anno_root}/pointedcap_mask3d_train_rephrase.json",
    ],
    'groundedqav2': [
        f"{anno_root}/groundedqa_{segmentor}_train{version}_rephrase.json",
    ],
    'scanreferv2': [
        f"{anno_root}/scanrefer_{segmentor}_train{version}_rephrase.json",
    ],
    'multi3drefv2': [
        f"{anno_root}/multi3dref_{segmentor}_train{version}_rephrase.json",
    ],
    'nr3dv2': [
        f"{anno_root}/nr3d_{segmentor}_train{version}_rephrase.json",
    ],
    'sr3d+v2': [
        f"{anno_root}/sr3d+_{segmentor}_train{version}_rephrase.json",
    ],
    'nr3d_captionv2': [
        f"{anno_root}/nr3d_caption_{segmentor}_train{version}_rephrase.json",
    ],
    'scanqav2': [
        f"{anno_root}/scanqa_train_rephrase.json",
    ],
    'sqa3dv2': [
        f"{anno_root}/sqa3d_train_rephrase.json",
    ],
    'partialref_scale': [
        f"{anno_root}/partialref_{segmentor}_train{version}_scale.json",
    ],
    'partial_objalign_scale': [
        f"{anno_root}/partial_obj_align_{segmentor}_train{version}_scale.json",
    ],
    'partial_od_scale': [
        f"{anno_root}/partial_od_{segmentor}_train{version}_scale.json",
    ],
}

# adversarial
# Hybrid Object Probing Evaluation (HOPE):
# partial_od_scale:
# [
#     {
#         "scene_id": "scene0191_00",
#         "obj_id": 0,
#         "prompt": "Can you find any salt; suitcases; trash can; door?",
#         "caption": "No; No; Yes. <OBJ000>; Yes. <OBJ004>."
#     },

# Hybrid Referring Object Classification (HROC): 
# partial_objalign_scale:
# [
#     {
#         "scene_id": "scene0191_00",
#         "obj_id": 0,
#         "prompt": "Is it correct that <OBJ001> is a chair; <OBJ000> is a trash can; <OBJ012> is a chair?",
#         "caption": "No, <OBJ001> is a backpack; Yes; No, <OBJ012> is a window."
#     },

# Partial Factual 3D Visual Grounding (PF3DVG): 
# partialref_scale:
# [
#     {
#         "scene_id": "scene0201_01",
#         "obj_id": 0,
#         "prompt": "If you can, please share the ID of the object that fits the description \"the picture that is close to the couch\".",
#         "caption": "No."
#     },
#     {
#         "scene_id": "scene0201_01",
#         "obj_id": 11,
#         "prompt": "If you can, please share the ID of the object that fits the description \"find the table that is over the couch\".",
#         "caption": "It is \"in front of\". <OBJ011>."
#     },
#     {
#         "scene_id": "scene0201_01",
#         "obj_id": 11,
#         "prompt": "If you can, please share the ID of the object that fits the description \"find the table that is in front of the couch\".",
#         "caption": "Yes. <OBJ011>."
# partialrefv2:
# [
#     {
#         "scene_id": "scene0201_01",
#         "obj_id": 11,
#         "prompt": "If you can, please share the ID of the object that fits the description \"Locate the table that is situated under the couch.\".",
#         "caption": "It is \"in front of\". <OBJ011>."
#     },

# Faithful 3D Question Answering (3DFQA): 
# groundedqa:
# [
#     {
#         "scene_id": "scene0000_00",
#         "obj_id": [
#             8
#         ],
#         "prompt": "What is in the right corner of room by curtains? If you can, answer the question using a word or a phrase. And provide all the IDs for objects related to the question and answer.",
#         "caption": "Brown cabinet with tv sitting in it. <OBJ032>."
#     },
# groundedqav2:
# [
#     {
#         "scene_id": "scene0000_00",
#         "obj_id": [
#             8
#         ],
#         "prompt": "What can be found in the right corner of the room next to the curtains? If you can, answer the question using a word or a phrase. And provide all the IDs for objects related to the question and answer.",
#         "caption": "Brown cabinet with tv sitting in it. <OBJ032>."
#     },

# diverse
# captioning: 
# nr3d_caption:
# [
#     {
#         "scene_id": "scene0525_00",
#         "obj_id": 9,
#         "prompt": "Detail the spatial positioning of the <OBJ067> amidst surrounding elements.",
#         "caption": "The plant at the far right hand side of the bookcase tucked in the furthest corner of the desk"
#     },
# scannet_caption:
# [
#     {
#         "scene_id": "scene0143_02",
#         "obj_id": 0,
#         "prompt": "Portray the visual characteristics of the <OBJ024>.",
#         "caption": "A dark wooden desk with rectangular shape and four legs."
#     },
# scannet_region_caption:
# [
#     {
#         "scene_id": "scene0191_00",
#         "obj_id": 2,
#         "prompt": "Describe the area surrounding <OBJ004>.",
#         "caption": "Adjacent to the wooden door (<OBJ004>) with a frosted glass panel, there is a frosted window (<OBJ006>) to the left and another clear window (<OBJ012>) further along the same wall to the right. Near the center of the room, a red and black backpack (<OBJ001>) is placed on a rectangular table with a light brown finish (<OBJ005>), while an office chair with a black cushioned seat (<OBJ008>) rests close to the backpack (<OBJ001>). Additionally, a blue trash can (<OBJ000>) with a black lid is situated closer to the entrance. A board with a metallic frame leaning against a wall (<OBJ011>) and a whiteboard with a metallic frame and wheels (<OBJ014>) are also next to the door (<OBJ004>)."
#     },
# nr3d_captionv2:
# [
#     {
#         "scene_id": "scene0525_00",
#         "obj_id": 9,
#         "prompt": "Detail the spatial positioning of the <OBJ067> amidst surrounding elements.",
#         "caption": "The plant positioned at the extreme right of the bookcase, nestled in the remotest corner of the desk"
#     },

# grounding: nr3d, sr3d+, scanreferv2, multi3drefv2, nr3dv2, sr3d+v2

# question answering: obj_align, scanqav2, sqa3dv2

# HOPE = 'partial_od_scale'
# HROC = 'partial_objalign_scale'
# PF3DVG = 'partialref_scale#partialrefv2'
# F3DFQA = 'groundedqa#groundedqav2'
# diverseCap = 'nr3d_caption#scannet_caption#scannet_region_caption#nr3d_captionv2'
# diverseVG = 'nr3d#sr3d+#scanreferv2#multi3drefv2#nr3dv2#sr3d+v2'
# diverseQA = 'obj_align#scanqav2#sqa3dv2'

# tasks = [HOPE, HROC, PF3DVG, F3DFQA]
# # tasks = [diverseCap, diverseVG, diverseQA]
# file = []
# for task in tasks:
#     for t in task.split('#'):
#         file.append(train_file_dict[t][0])
# adversarial_question_answering = []


def clean_sentence(sentence):
    # 正则表达式匹配 (<OBJ数字>) 或 <OBJ数字>
    pattern = r'\(<OBJ\d+>\)|<OBJ\d+>'
    # 使用 re.sub 替换匹配的部分为空字符串
    sentence = re.sub(pattern, '', sentence)
    sentence = re.sub('ids', '', sentence)
    sentence = re.sub('IDs', '', sentence)
    sentence = re.sub('ID', '', sentence)
    sentence = re.sub('id', '', sentence)
    return sentence

# for f in tqdm(file):
#     data = json.load(open(f, 'r'))
#     for d in data:
#         # 使用 re.sub 替换匹配的部分为空字符串
#         prompt = clean_sentence(d['prompt'])
#         caption = clean_sentence(d['caption'])
#         adversarial_question_answering.append(prompt)
#         adversarial_question_answering.append(caption)


# # 合并所有文本内容
# combined_text = ' '.join(adversarial_question_answering)

# # 定义生成词云的函数
# def generate_wordcloud(text, title):
#     wordcloud = WordCloud(
#         background_color='white', 
#         max_words=10000, 
#         collocations=False, 
#         width=800, 
#         height=400
#     ).generate(text)

#     # 保存词云为图像文件
#     # plt.figure(figsize=(10, 5))
#     # plt.imshow(wordcloud) #, interpolation='bilinear')
#     # plt.axis('off')
#     # plt.savefig(f'/home/kangweitai/3DLLM/Chat-3D-v2/{title}_wordcloud.pdf', format='pdf', dpi=1200)
#     # plt.close()

#     # 保存词云为矢量图（PDF格式）
#     # wordcloud_svg = wordcloud.to_svg(embed_font=True)  # 生成矢量 SVG 数据
#     # svg_io = BytesIO(wordcloud_svg.encode('utf-8'))  # 将 SVG 转为字节流
#     # # 使用 reportlab 将 SVG 保存为 PDF 矢量文件
#     # drawing = svg2rlg(svg_io)  # 将字节流加载为矢量对象
#     # renderPDF.drawToFile(drawing, f'/home/kangweitai/3DLLM/Chat-3D-v2/{title}_wordcloud.pdf')

#     # 生成词云的 SVG 矢量内容
#     svg_content = wordcloud.to_svg(embed_font=True)
#     # 保存为 .svg 文件
#     with open(f'/home/kangweitai/3DLLM/Chat-3D-v2/{title}_wordcloud.svg', 'w', encoding='utf-8') as svg_file:
#         svg_file.write(svg_content)

# # 生成并保存词云
# generate_wordcloud(combined_text, 'adversarial')

# ------------------- 长度统计 箱型图 ------------------- #

import seaborn as sns

HOPE = 'partial_od_scale'
HROC = 'partial_objalign_scale'
PF3DVG = 'partialref_scale#partialrefv2'
F3DFQA = 'groundedqa#groundedqav2'
diverseCap = 'nr3d_caption#scannet_caption#scannet_region_caption#nr3d_captionv2'
diverseVG = 'nr3d#sr3d+#scanreferv2#multi3drefv2#nr3dv2#sr3d+v2'
diverseQA = 'obj_align#scanqav2#sqa3dv2'
benmark = "scanrefer#multi3dref#scan2cap#scanqa#sqa3d"

tasks = [HOPE, HROC, PF3DVG, F3DFQA] + [diverseCap, diverseVG, diverseQA] + [benmark]
tasks_name = ['Hybrid Object Probing Evaluation', 'Hybrid Referring Object Classification', 'Partial Factual 3D Visual Grounding', 'Faithful 3D Question Answering', 
              'Diverse Captioning', 'Diverse Visual Grounding', 'Diverse Question Answering', 'Benchmark Data']
# for task in tasks:
#     num_data = 0
#     file = []
#     for t in task.split('#'):
#         file.append(train_file_dict[t][0])
#     for f in file:
#         data = json.load(open(f, 'r'))
#         num_data += len(data)

# 收集数据长度
task_sentence_lengths = {}
for i, task in tqdm(enumerate(tasks), total=len(tasks)):
    task_name = tasks_name[i]
    task_sentence_lengths[task_name] = []
    for t in task.split('#'):
        with open(train_file_dict[t][0], 'r') as f:
            data = json.load(f)
            for item in data:
                prompt = clean_sentence(item['prompt']).split()
                caption = clean_sentence(item['caption']).split()
                task_sentence_lengths[task_name].append(len(prompt) + len(caption))
# 计算每个任务的平均句子长度
average_lengths = {task: (sum(lengths) / len(lengths) if lengths else 0) for task, lengths in task_sentence_lengths.items()}

from matplotlib.cm import get_cmap

# 计算平均长度后排序（从小到大）
sorted_average_lengths = dict(sorted(average_lengths.items(), key=lambda x: x[1], reverse=True))

# 设置颜色，使用渐变色或不同颜色
cmap = get_cmap("tab20")
colors = [cmap(i % 20) for i in range(len(sorted_average_lengths))]

# 绘制直方图
plt.figure(figsize=(12, 8))
bars = plt.barh(
    list(sorted_average_lengths.keys()), 
    list(sorted_average_lengths.values()), 
    color=colors
)

# 添加虚线刻度
plt.grid(axis="x", linestyle="--", alpha=0.6)

# 在每个柱子顶端标上数值
for bar in bars:
    plt.text(
        bar.get_width() + 0.2,  # 调整偏移量
        bar.get_y() + bar.get_height() / 2, 
        f"{bar.get_width():.1f}",
        va="center", 
        fontsize=10
    )

# 设置标题和标签
plt.xlabel("Average Sentence Length", weight="bold")
plt.yticks(weight="bold")
# 保存为高分辨率 PDF
output_path = "/home/kangweitai/3DLLM/Chat-3D-v2/Box_plot.pdf"
plt.tight_layout()
plt.savefig(output_path, format="pdf", dpi=1200)
plt.close()