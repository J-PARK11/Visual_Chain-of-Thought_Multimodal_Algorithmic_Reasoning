import base64
import json
import io
import os
from datasets import load_dataset
import math
import pandas as pd
from PIL import Image
import glob

# 1. 데이터셋 불러오기


def generate_caption():
    base_directory = f'/home/work/g-earth-22/VLM/VLM/database/MMBench/captions'
    json_files = glob.glob(os.path.join(base_directory, '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)

    return caption_dict

def load_image(image_path: str):
    return Image.open(image_path).convert("RGB")

def decode_base64_to_image(base64_string):
    image_data = base64.b64decode(base64_string)
    image = load_image(io.BytesIO(image_data))
    return image

def generate_mmbench_qainfo(filee):
    dataset = load_dataset(filee)#"HuggingFaceM4/MMBench_dev"
    df = dataset['train']
    no_img = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']
    caption_dict = generate_caption()
    qa_dict = []
    num_data = len(df)
    count = 0
    for i in range(num_data): # 4329 for DEV_EN
        try:
            datum = df[i]
            single_dict = {}
            single_dict['id'] = i+1
            image = datum["image"]
            single_dict['image'] = image
            single_dict['puzzle_id'] = '10' #일단 하드코딩
            single_dict['Question'] = datum['question'].strip()
            single_dict['A'] = datum['A']
            single_dict['B'] = datum['B']
            single_dict['C'] = datum['C']
            single_dict['D'] = datum['D']
            for opt in ['A', 'B', 'C', 'D', 'E']:
                try:
                    if math.isnan(single_dict[opt]) == True:
                        del single_dict[opt]
                except:
                    continue
            single_dict['Answer'] = chr(ord('A') + datum['label'])
            single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
            qa_dict.append(single_dict)
            if os.path.isfile(os.path.join(data_dir, 'SAM_features', f'{i}.npy')):
                single_dict['sam_feature_path'] = os.path.join(data_dir, 'SAM_features', f'{i}.npy') # todo
                single_dict['caption'] = caption_dict[str(i)]['caption']
            else:
                count += 1
        except:
            count += 1
    print(f'mmbench / {count}')

    return qa_dict
    
# mmbench_qa_dict = generate_mmbench_qainfo(df)
# print('mmbench data num :', len(mmbench_qa_dict))
# print (len(mmbench_qa_dict))
# print (mmbench_qa_dict[0])
# print (mmbench_qa_dict[1])
# print (mmbench_qa_dict[2])
# print (mmbench_qa_dict[100])