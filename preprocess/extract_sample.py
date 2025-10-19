import os
import json
import random
import shutil
from tqdm import tqdm

# def process_json_files(input_folder, output_folder):
#     # 确保输出文件夹存在
#     os.makedirs(output_folder, exist_ok=True)
    
#     # 遍历输入文件夹中的所有文件
#     for filename in os.listdir(input_folder):
#         if filename.endswith('.json'):
#             input_file_path = os.path.join(input_folder, filename)
#             output_file_path = os.path.join(output_folder, filename)
            
#             with open(input_file_path, 'r') as f:
#                 data = json.load(f)
            
#             # 检查文件名是否包含“train”
#             if 'train' in filename:
#                 # 随机抽取0.3 * len(list)的元素
#                 sample_size = int(0.3 * len(data))
#                 sampled_data = random.sample(data, sample_size)
#                 print(f'Sampled {sample_size} annotations from {filename}')
#                 with open(output_file_path, 'w') as f:
#                     json.dump(sampled_data, f, indent=4)
#             else:
#                 # 原封不动地保存到输出文件夹
#                 shutil.copy(input_file_path, output_file_path)

# # 调用函数处理指定文件夹中的json文件
# input_folder = '/data/kangweitai/3D/chat3d-anno/chat3d-anno-iou50-oneprompt/'  # 输入文件夹路径
# output_folder = '/data/kangweitai/3D/chat3d-anno/chat3d-anno-iou50-oneprompt-0.3/'  # 输出文件夹路径

# process_json_files(input_folder, output_folder)


# 读取json文件
dir = '/data/kangweitai/3D/chat3d-anno/mask3d150sort-uni3d-dinov2Giant-iou50-oneprompt/'
file_names = ['scan2cap_mask3d_train.json', 'sqa3d_train.json', 'scanqa_train.json']
save_names = ['scan2cap_mask3d_train_scale2.json', 'sqa3d_train_scale2.json', 'scanqa_train_scale2.json']
for file_name in file_names:
    annos = json.load(open(os.path.join(dir, file_name), 'r'))
    new_annos = []
    if 'cap' in file_name:
        new_annos += annos * 2
    else:
        new_annos += annos * 2
        for anno in tqdm(annos, total=len(annos)):
            if len(anno["caption"].split()) > 3:
                new_annos += [anno] * 2
    print(f'--- ori {file_name} has {len(annos)}')
    print(f'--- {len(new_annos)} in {file_name}')
    
    with open(os.path.join(dir, save_names[file_names.index(file_name)]), 'w') as f:
        json.dump(new_annos, f, indent=4)