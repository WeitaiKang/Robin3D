import json
import re
from tqdm import tqdm

# refer to LEO: embodied-generalist
# https://github.com/embodied-generalist/embodied-generalist/blob/477dc44b8b18dbfbe6823c307436d896ec8b062e/data/data_utils.py#L322-L379
def clean_answer(data):
    data = re.sub('[ ]+$' ,'', data)
    data = re.sub('^[ ]+' ,'', data)
    data = re.sub(' {2,}', ' ', data)

    data = re.sub('\.[ ]{2,}', '. ', data)
    data = re.sub('ç' ,'c', data)
    data = re.sub('’' ,'\'', data)
    data = re.sub(r'\bletf\b' ,'left', data)
    data = re.sub(r'\blet\b' ,'left', data)
    data = re.sub(r'\btehre\b' ,'there', data)
    data = re.sub(r'\brigth\b' ,'right', data)
    data = re.sub(r'\brght\b' ,'right', data)
    data = re.sub(r'\bbehine\b', 'behind', data)
    data = re.sub(r'\bchai\b' ,'chair', data)
    data = re.sub(r'\bwasing\b' ,'washing', data)
    data = re.sub(r'\bwaslked\b' ,'walked', data)
    data = re.sub(r'\boclock\b' ,'o\'clock', data)
    data = re.sub(r'\bo\'[ ]+clock\b' ,'o\'clock', data)

    # digit to word, only for answer
    data = re.sub(r'\b0\b', 'zero', data)
    data = re.sub(r'\bnone\b', 'zero', data)
    data = re.sub(r'\b1\b', 'one', data)
    data = re.sub(r'\b2\b', 'two', data)
    data = re.sub(r'\b3\b', 'three', data)
    data = re.sub(r'\b4\b', 'four', data)
    data = re.sub(r'\b5\b', 'five', data)
    data = re.sub(r'\b6\b', 'six', data)
    data = re.sub(r'\b7\b', 'seven', data)
    data = re.sub(r'\b8\b', 'eight', data)
    data = re.sub(r'\b9\b', 'nine', data)
    data = re.sub(r'\b10\b', 'ten', data)
    data = re.sub(r'\b11\b', 'eleven', data)
    data = re.sub(r'\b12\b', 'twelve', data)
    data = re.sub(r'\b13\b', 'thirteen', data)
    data = re.sub(r'\b14\b', 'fourteen', data)
    data = re.sub(r'\b15\b', 'fifteen', data)
    data = re.sub(r'\b16\b', 'sixteen', data)
    data = re.sub(r'\b17\b', 'seventeen', data)
    data = re.sub(r'\b18\b', 'eighteen', data)
    data = re.sub(r'\b19\b', 'nineteen', data)
    data = re.sub(r'\b20\b', 'twenty', data)
    data = re.sub(r'\b23\b', 'twenty-three', data)

    data = re.sub(r'\bbackwards\b', 'backward', data)

    # data = data[0].upper() + data[1:]

    return data

for split in ['train', 'val']:
    scanrefer = json.load(open(f'/data/kangweitai/3D/annotation/grounding/ScanRefer/ScanRefer_filtered_{split}.json'))

    clean_answer_num = 0
    clean_prompt_num = 0
    for anno in tqdm(scanrefer):
        clean_caption = clean_answer(anno['description'])
        if clean_caption != anno['description']:
            clean_answer_num += 1
            # print(f'--- before: {anno["caption"]} ---')
            # print(f'--- after: {clean_caption} ---')
        anno['description'] = clean_caption

        # clean_prompt = clean_answer(anno['prompt'])
        # if clean_prompt != anno['prompt']:
        #     clean_prompt_num += 1
        #     # print(f'--- before: {anno["prompt"]} ---')
        #     # print(f'--- after: {clean_prompt} ---')
        # anno['prompt'] = clean_prompt

    print(f'--- clean {clean_answer_num} answers & {clean_prompt_num} prompts of {len(scanrefer)} annotations ---')

    with open(f'/data/kangweitai/3D/clean-annotation/ScanRefer_filtered_{split}.json', 'w') as f:
        json.dump(scanrefer, f, indent=4)