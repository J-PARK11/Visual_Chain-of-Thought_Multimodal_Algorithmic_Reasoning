import base64
import json
import io
import os
import math
import pandas as pd
from PIL import Image
import glob


from datasets import load_dataset, concatenate_datasets




def generate_caption(): # todo
    base_directory = '/home/work/g-earth-22/VLM/VLM/database/MMMU'
    json_files = glob.glob(os.path.join(base_directory, 'captions', '*', '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)
            # print('len', len(caption_dict))

    return caption_dict


def count_num_img(datum):
    n_img = 0

    for i in range(7):
        key = 'image_%s'%(i+1)
        if datum[key] != None:
            n_img += 1
    return n_img


def generate_mmmu_qainfo(filee):
    # 모든 분야 리스트
    fields = ['Accounting', 'Agriculture', 'Architecture_and_Engineering', 'Art', 'Art_Theory', 'Basic_Medical_Science', 'Biology', 'Chemistry', 'Clinical_Medicine', 'Computer_Science', 'Design', 'Diagnostics_and_Laboratory_Medicine', 'Economics', 'Electronics', 'Energy_and_Power', 'Finance', 'Geography', 'History', 'Literature', 'Manage', 'Marketing', 'Materials', 'Math', 'Mechanical_Engineering', 'Music', 'Pharmacy', 'Physics', 'Psychology', 'Public_Health', 'Sociology']

    # 모든 데이터셋을 저장할 리스트
    all_datasets = []

    # 각 분야별로 데이터셋을 불러와 합치기
    for field in fields:
        dataset = load_dataset(filee, field) #"MMMU/MMMU"
        combined = concatenate_datasets([dataset['dev'], dataset['validation'], dataset['test']])
        all_datasets.append(combined)

    # 모든 분야의 데이터셋을 하나로 합치기
    df = concatenate_datasets(all_datasets)


    qtype_count = 0
    mc_qtype_count = 0
    max_img_count = 0
    max_opt_count = 0
    # option = f"{file.split(base_path)[1].split('/')[0]}"
    caption_dict = generate_caption()
    opt_char = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H']

    qa_dict = []
    num_data = len(df)
    count = 0

    for i in range(num_data):
        try:
            datum = df[i]

            if datum['question_type'] != 'multiple-choice':
                qtype_count += 1
                continue

            else: # multiple-choice samples: 
                mc_qtype_count += 1

                single_dict = {}
                single_dict['id'] = datum['id']

                n_img = count_num_img(datum)
                if n_img != 1:
                    max_img_count += 1
                    continue
                else:
                    single_dict['image'] = datum['image_1']

                single_dict['Question'] = datum['question'].strip()
                options = eval(datum['options'])

                try: 
                    for j, opt in enumerate(options):
                        single_dict[opt_char[j]] = opt
                except:
                    max_opt_count += 1
                    continue

                single_dict['Answer'] = datum['answer']
                single_dict['AnswerValue'] = single_dict[single_dict['Answer']]
                single_dict['puzzle_id'] = '10' #일단 하드코딩
                cur_option = '_'.join(datum['id'].split('_')[1:-1])
                cur_idx = datum['id'].split('_')[-1]
                qa_dict.append(single_dict)
                if os.path.isfile(os.path.join(base_path, 'SAM_features', cur_option, f'{cur_idx}.npy')):
                    single_dict['sam_feature_path'] = os.path.join(base_path, 'SAM_features', cur_option, f'{cur_idx}.npy') # todo
                    single_dict['caption'] = caption_dict[datum['id']]['caption']
                else:
                    count+=1
        except:
            count+=1
    print(f'mmmu / {count}')

    # print (qtype_count, mc_qtype_count, max_img_count, max_opt_count) 
    return qa_dict

# mmmu_qa_dict = generate_mmmu_qainfo(data_sets_all)

# for k, file in enumerate(files):
#     df = pd.read_parquet(file)
#     if k == 1:
#         mmmu_qa_dict = generate_mmmu_qainfo(df, file)
#     else:
#         mmmu_qa_dict += generate_mmmu_qainfo(df, file)
# print('mmmu data num :', len(mmmu_qa_dict))
# print (len(mmmu_qa_dict))
# print (mmmu_qa_dict[0])
# print (mmmu_qa_dict[1])
# print (mmmu_qa_dict[2])
# print (mmmu_qa_dict[100])