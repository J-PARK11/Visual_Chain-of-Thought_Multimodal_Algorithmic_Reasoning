from datasets import load_dataset
import pandas as pd
import json
import os
import glob

# 1. 데이터셋 불러오기

def generate_caption():
    base_directory = '/home/work/g-earth-22/VLM/VLM/database/MathVista/captions'
    json_files = glob.glob(os.path.join(base_directory, '*.json'), recursive=True)
    caption_dict = dict()

    for json_file in json_files:
        with open(json_file, "r") as tmp:
            caption = json.load(tmp)
            caption_dict.update(caption)

    return caption_dict


def load_jsonl_with_pandas(filename):
    data = []
    with open(filename, 'r') as f:
        for line in f:
            line = line.strip()
            if line:
                try:
                    data.append(json.loads(line))
                except json.JSONDecodeError as e:
                    print(f"Error decoding JSON on line: {line}")
                    print(f"Error: {e}")
    df = pd.DataFrame(data)
    return df

data_dir = '/data/SMART101/MathVista'
img_dir = '/data/SMART101/MathVista/'


def generate_mathvista_qainfo(filee):
    dataset = load_dataset(filee) #"AI4Math/MathVista"
    df = dataset['testmini']
    count = 0
    mc_count = 0
    max_opt_count = 0
    no_ques = 0
    invalid_ques = 0
    invalid_opts = 0
    invalid_ans = 0
    opt_char = ['A', 'B', 'C', 'D', 'E']
    caption_dict = generate_caption()
    qa_dict = []
    count = 0
    for i in range(1, 1001): # total 1000 samples
        try:
            datum = df[i]
            if datum['metadata']['split'] == 'test':
                continue # no answer

            if datum['question_type'] != 'multi_choice': # skip free-form samples
                count += 1 # 460 samples
                continue

            else: # multiple-choice samples: 540 samples
                if len(datum['choices']) > 5:
                    max_opt_count += 1 # 13 samples
                    continue

                mc_count += 1

                single_dict = {}
                single_dict['id'] = datum['pid']
                single_dict['image'] = os.path.join(img_dir, datum['image'])
                single_dict['puzzle_id'] = '10' #일단 하드코딩
                single_dict['Question'] = datum['question'].strip()
                for j, choice in enumerate(datum['choices']):
                    single_dict[['A', 'B', 'C', 'D', 'E'][j]] = choice

                for j, choice in enumerate(datum['choices']):
                    if datum['answer'] == choice:
                        single_dict['Answer'] = ['A', 'B', 'C', 'D', 'E'][j]
                single_dict['AnswerValue'] = datum['answer']
                qa_dict.append(single_dict)
                if os.path.isfile(os.path.join(data_dir, 'SAM_features', f'{i}.jpg.npy')):
                    single_dict['sam_feature_path'] = os.path.join(data_dir, 'SAM_features', f'{i}.jpg.npy')
                    single_dict['caption'] = caption_dict[datum['image'].split('/')[1]]['caption']
                else:
                    count += 1
        except:
            count += 1
    print(f'mathvista / {count}')

    
    # print (mc_count, count, max_opt_count) 
    return qa_dict
    
# mathvista_qa_dict = generate_mathvista_qainfo(df)
# print('mathvista data num :', len(mathvista_qa_dict))

# print (len(mathvista_qa_dict))
# print (mathvista_qa_dict[0])
# print (mathvista_qa_dict[1])
# print (mathvista_qa_dict[2])
# print (mathvista_qa_dict[100])