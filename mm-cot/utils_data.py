
import os
import json
import torch
import random
import numpy as np
from utils_prompt import *
from torch.utils.data import Dataset
from lib.utils_smart import *
import lib.globvars_smart as gv



img_shape = {
    "resnet": (512, 2048),
    "clip": (49, 2048),
    "detr": (100, 256),
    "vit": (145, 1024),
}

def load_img_features(args):
    
    data_type = args.data_type
    if data_type == 'scienceqa':
        problems = json.load(open(os.path.join('/data/ScienceQA/', 'problems.json')))
        pid_splits = json.load(open(os.path.join('/data/ScienceQA/', 'pid_splits.json')))
        captions = json.load(open(f'./data/vision_features/{data_type}/instruct_captions.json'))["captions"]
        
        for qid in problems:
            problems[qid]['caption'] = captions[qid] if qid in captions else ""
        
        train_qids = pid_splits['%s' % ('train')]
        val_qids = pid_splits['%s' % ('val')]
        test_qids = pid_splits['%s' % ('test')]
        
        qids = {'train': train_qids, 'val':val_qids,'test':test_qids}
    else:
        problems, qids = None, None
    
    with open(os.path.join(f'./data/vision_features/{data_type}/', 'name_map.json'), 'r') as infile:
        name_maps = json.load(infile)
    if args.img_type == "resnet":
        image_features = np.load(f'./data/vision_features/{data_type}/resnet.npy')
        image_features = np.expand_dims(image_features, axis=1)
        image_features = image_features.repeat(512, axis=1)
    elif args.img_type == "clip":
        image_features = np.load(f'./data/vision_features/{data_type}/clip.npy')
    elif args.img_type == "detr":
        image_features = torch.load(f"./data/vision_features/{data_type}/detr.pth")
        # image_features = np.load('./data/vision_features/smart/detr.npy')
    elif args.img_type == "vit":
        image_features = torch.load(f"./data/vision_features/{data_type}/vit.pth")
    else:
        image_features = np.load(f'./data/vision_features/{data_type}/detr.npy')

    print(f"{args.img_type} img_features size: {image_features.shape}\n")
    
    return name_maps, image_features, problems, qids

class MMCOT_SMART_Dataset(Dataset):

    def __init__(
        self, args, name_maps, image_features, tokenizer, mode
    ):
        
        self.qa_info = []
        self.args = args
        self.mode = mode
        self.diff = 'easy'
        self.data_root = args.data_root
        self.tokenizer = tokenizer
        self.source_len = args.input_len
        self.summ_len = args.output_len
        puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
        self.name_maps = name_maps
        self.image_ids = image_features
        # self.image_token_id = tokenizer.additional_special_tokens_ids[
            # tokenizer.additional_special_tokens.index("<image>")]
        
        if self.mode == 'train':
            self.num_tot = 1000
        elif self.mode == 'extract':
            self.num_tot = 2000
        elif self.mode in ['valid', 'gen_valid']:
            self.num_tot = 1
        elif self.mode in ['test', 'gen_test']:
            self.num_tot = 300
        
        # 시퀀스 퍼즐 예외처리.
        """
        seq_puzzle = list(map(str, gv.SEQ_PUZZLES))
        puzzle_ids = list(set(puzzle_ids) - set(seq_puzzle))        
        """
        
        # GPT로 증강된 GT Rationale 가져오기
        if self.mode == 'train':
            self.level2_puzzle_list = ['2','19','23','28','44','46','50','56','89']
            self.level3_puzzle_list = ['13','16','17','24','39','40','43','51','54','58','80','93']
            
            if args.gpt_data_include_level == 2:
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level3_puzzle_list)))
            elif args.gpt_data_include_level == 1:
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level2_puzzle_list)))
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level3_puzzle_list)))
        
        if self.mode != 'extract':
            self.GPT_augmentation_dict_path = args.GPT_augmentation_dict_path
            with open(self.GPT_augmentation_dict_path,'r') as f:
                self.GPT_augmentation_dict = json.load(f)
        
        # 인스턴스 퍼즐 불러오기.
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            
            if self.mode in ['train']:
                usuable_info = []
                qa_info = qa_info[:self.num_tot]
                for ub_id, ub_sample in enumerate(qa_info):
                    im_name = ub_sample['image']
                    option_answer = f'Answer: {ub_sample["Answer"]}'
                    rationale = self.GPT_augmentation_dict[im_name]['GT_with_Rationale']
                    if (len(rationale) < 1000) and (option_answer in rationale):
                        usuable_info.append(ub_sample)
                qa_info = usuable_info
            elif self.mode in ['test', 'gen_test']:
                qa_info = qa_info[1700:1700+self.num_tot]
            else: # valid
                qa_info = qa_info[:self.num_tot]
                
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = get_val(qa_info[t], qa_info[t]["Answer"])
            
            self.qa_info = self.qa_info + qa_info
        
        # 학습 데이터는 셔플.
        if self.mode == 'train':
            random.seed(1123)
            random.shuffle(self.qa_info)
            print('Train Dataset shuffled')        
        
        print(f'{self.mode} Dataloader: {len(self.qa_info)}')
    
    def __len__(self):
        return len(self.qa_info)
    
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, index):
        info = self.qa_info[index]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im_name = im_path.split('/')[-1] 
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        seq_answer = np.zeros(gv.MAX_DECODE_STEPS)
        
        if self.mode in ['extract', 'gen_valid', 'gen_test']:
            return im_path, im_name, info["Question"]
        
        # 시퀀스 데이터 예외처리.
        if int(pid) not in gv.SEQ_PUZZLES:
            seq_answer[0] = answer_value
        else:
            try:
                seq_answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()     
        
        # 옵션 프롬프팅 및 질문과 풀이과정 정의
        opts = []
        Answer_Option_phrase = '\nOptions:'
        for op in ["A", "B", "C", "D", "E"]:
            op_val = info[op]
            Answer_Option_phrase += f' {op}={op_val}'
            opts.append(op_val)
        option_answer= f'Answer: {info["Answer"]}' # info['Answer']
        question = info["Question"]
        question = question + Answer_Option_phrase
        
        if self.mode in ['train', 'valid']:
            rationale = self.GPT_augmentation_dict[im_name]['GT_with_Rationale']
            rationale = rationale[:rationale.find('Answer')]

            source_text = f"Question: {question}'\nLook at the image and the question, Please explain the solution and rationale for solving the question.'"
            target_text = f"Solution: {rationale}"
        
        else: # test
            source_text = f"Question: {question}'\nLook at the image and the question, Please explain the solution and rationale for solving the question.'"
            target_text = f"Solution: {option_answer}"

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()        
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = self.image_ids[int(self.name_maps[im_name])]
        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids}


class ScienceQADatasetImg(Dataset):
    """
    Creating a custom dataset for reading the dataset and
    loading it into the dataloader to pass it to the
    neural network for finetuning the model

    """

    def __init__(
        self, problems, qids, name_maps, tokenizer, source_len, target_len, args, image_features, test_le=None, mode=None
    ):
        """
        Initializes a Dataset class

        Args:
            dataframe (pandas.DataFrame): Input dataframe
            tokenizer (transformers.tokenizer): Transformers tokenizer
            source_len (int): Max length of source text
            target_len (int): Max length of target text
            source_text (str): column name of source text
            target_text (str): column name of target text
        """
        self.tokenizer = tokenizer
        self.data = {qid : problems[qid] for qid in qids}
        self.source_len = source_len
        self.summ_len = target_len
        self.target_text = []
        self.source_text = []
        self.image_ids = []
        if test_le is not None:
            test_le_data =json.load(open(test_le))["preds"]
        else:
            test_le_data = None
        idx = 0
        for qid in self.data:
            if test_le_data is not None:
                curr_le_data = test_le_data[idx]
                idx += 1
            else:
                curr_le_data = None
            prompt, target = build_train_pair(problems, qid, args, curr_le_data)
            self.target_text.append(target)
            self.source_text.append(prompt)
            if str(qid) in name_maps:
                i_vectors = image_features[int(name_maps[str(qid)])]
                self.image_ids.append(i_vectors)
            else:
                shape = img_shape[args.img_type]
                self.image_ids.append(np.zeros(shape))
    
    def __len__(self):
        """returns the length of dataframe"""

        return len(self.target_text)

    def __getitem__(self, index):
        """return the input ids, attention masks and target ids"""

        source_text = str(self.source_text[index])
        target_text = str(self.target_text[index])
        image_ids = self.image_ids[index]

        # cleaning data so as to ensure data is in string type
        source_text = " ".join(source_text.split())
        target_text = " ".join(target_text.split())

        source = self.tokenizer.batch_encode_plus(
            [source_text],
            max_length=self.source_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        target = self.tokenizer.batch_encode_plus(
            [target_text],
            max_length=self.summ_len,
            pad_to_max_length=True,
            truncation=True,
            padding="max_length",
            return_tensors="pt",
        )
        source_ids = source["input_ids"].squeeze()
        source_mask = source["attention_mask"].squeeze()
        target_ids = target["input_ids"].squeeze().tolist()

        image_ids = torch.tensor(image_ids).squeeze()
        
        return {
            "input_ids": source_ids,
            "attention_mask": source_mask,
            "image_ids": image_ids,
            "labels": target_ids,
        }
            
