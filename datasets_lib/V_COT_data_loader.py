import os
import pdb
import math
import json
import torch
import random
import pickle
import warnings
import numpy as np
from PIL import Image
warnings.filterwarnings("ignore")

from torch.utils.data import Dataset
from transformers.image_utils import load_image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import lib.utils as utils
import lib.V_COT_globvars as gv


class V_COT_SMART101_Dataset(Dataset):
    def __init__(self, args, mode):

        self.mode = mode
        self.diff = 'easy'
        self.data_root = args.data_root
        
        if self.mode == 'train':
            if args.task == 'custom':
                self.puzzle_list = args.train_puzzle_list
                puzzle_ids = self.puzzle_list.split(',')             
            elif args.task in ['supervised', 'GT_with_rationale', 'GPT_paraphrasing', 'GPT_augmentation_train']:
                puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
            elif args.task == 'GPT_augmentation_generation':
                puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
                phase, phase_num = int(args.phase[0]), int(args.phase[-1])
                unit = math.ceil(len(puzzle_ids) / phase_num)
                start_idx, end_idx = unit*(phase-1), unit*phase
                puzzle_ids = puzzle_ids[start_idx:end_idx]
            elif args.task == 'zero_shot':
                puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
                puzzle_ids = sorted(list(set(puzzle_ids) - set(['1','2','6','7','17','19','40','77'])))
                puzzle_ids = self.puzzle_list.split(',')
            self.num_tot = args.train_tot
        
        if self.mode == 'valid':
            self.puzzle_list = args.val_puzzle_list
            puzzle_ids = self.puzzle_list.split(',')
            self.num_tot = args.eval_tot
            
        elif self.mode == 'test':
            self.puzzle_list = args.test_puzzle_list
            puzzle_ids = self.puzzle_list.split(',')
            self.num_tot = args.eval_tot
        
        self.qa_info = []
        self.args = args
        self.task = args.task
        
        # 시퀀스 퍼즐 예외처리.
        """
        seq_puzzle = list(map(str, gv.SEQ_PUZZLES))
        puzzle_ids = list(set(puzzle_ids) - set(seq_puzzle))        
        """
        
        # GT with Rationale
        if args.task in ['GT_with_rationale', 'GPT_augmentation_generation']:
            self.GT_with_rationale_dict_path = args.GT_with_rationale_dict_path
            with open(self.GT_with_rationale_dict_path,'r') as f:
                self.GT_with_rationale = json.load(f)
                self.GT_with_rationale_key_list = dict()
        
        # GPT Paraphrasing
        if args.task == 'GPT_paraphrasing':
            self.GPT_paraphrasing_dict_path = args.GPT_paraphrasing_dict_path
            with open(self.GPT_paraphrasing_dict_path,'r') as f:
                self.GPT_paraphrasing_dict = json.load(f)        
          
        # GPT Augmentation Train
        if args.task == 'GPT_augmentation_train':
            
            self.level2_puzzle_list = ['2','19','23','28','44','46','50','56','89']
            self.level3_puzzle_list = ['13','16','17','24','39','40','43','51','54','58','80','93']
            
            if args.gpt_data_include_level == 2:
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level3_puzzle_list)))
            elif args.gpt_data_include_level == 1:
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level2_puzzle_list)))
                puzzle_ids = sorted(list(set(puzzle_ids) - set(self.level3_puzzle_list)))
            
            self.GPT_augmentation_dict_path = args.GPT_augmentation_dict_path
            with open(self.GPT_augmentation_dict_path,'r') as f:
                self.GPT_augmentation_dict = json.load(f)
                # augment_puzzle_name = list(self.GPT_augmentation_dict.keys())
        
        # 인스턴스 퍼즐 불러오기.
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            
            if self.mode == 'train' and args.task == 'supervised':
                qa_info = qa_info[args.eval_tot: self.num_tot+args.eval_tot]
            elif self.mode == 'valid' and args.task == 'GT_with_rationale':
                qa_info = qa_info[1: self.num_tot+1]
            elif self.mode == 'train' and args.task == 'GPT_paraphrasing':
                qa_info = [qa_info[0], qa_info[0], qa_info[0], qa_info[0]]
                for parapharsing_loop in range(4):
                    qa_info[parapharsing_loop]['train_seq'] = parapharsing_loop
            elif self.mode == 'train' and args.task == 'GPT_augmentation_train':
                qa_info = qa_info[1:1001]
                # matched_info = []
                # for search_info in qa_info:
                #     if search_info['image'] in augment_puzzle_name:
                #         matched_info += [search_info]
                # qa_info = matched_info
            elif args.task == 'GPT_augmentation_generation':
                self.GT_with_rationale_key_list[puzzle_id] = qa_info[0]['image']
                qa_info = qa_info[1: 1+self.num_tot]
            elif self.mode == 'test':
                qa_info = qa_info[1700:1700+self.num_tot]
            else:
                qa_info = qa_info[: self.num_tot]
                
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + qa_info
        
        # 학습 데이터는 셔플.
        if self.mode == 'train' and args.task != 'GPT_augmentation_generation':
            random.seed(1123)
            random.shuffle(self.qa_info)
            print('Train Dataset shuffled')
            
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        im_path = gv.osp(self.data_root, puzzle_root, "img", info["image"])
        im_name = im_path.split('/')[-1] 
        im = load_image(im_path)
        # im = Image.open(im_path)
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = np.zeros(gv.MAX_DECODE_STEPS)
        
        # 시퀀스 데이터 예외처리.
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()
        
        if not info: info = []
        opts = []
        Answer_Option_phrase = '\nOptions:'
        for op in ["A", "B", "C", "D", "E"]:
            op_val = info[op]
            Answer_Option_phrase += f'\n{op}. {op_val}'
            opts.append(op_val)
        q_stn = info["Question"]
        q_stn_out = q_stn + Answer_Option_phrase + '\nPlease answer with the alphabet in the options.'
        
        option_answer= f'Answer: {info["Answer"]}' # info['Answer']
        
        if self.task=='GT_with_rationale' and self.mode == 'train':
            option_answer = self.GT_with_rationale[im_name]['GT_with_Rationale']
        elif self.task=='GPT_augmentation_generation' and self.mode == 'train':
            ref_puzzle_name = self.GT_with_rationale_key_list[pid]
            option_answer = self.GT_with_rationale[ref_puzzle_name]['GT_with_Rationale']  
        elif self.task=='GPT_augmentation_train' and self.mode == 'train':
            rationale = self.GPT_augmentation_dict[im_name]['GT_with_Rationale']
            if len(rationale) < 1000:
                option_answer = rationale
                q_stn_out = q_stn + Answer_Option_phrase + '\nPlease answer with the alphabet in the options and explain how you solved it.'
            else:
                pass
        elif self.task =='GPT_paraphrasing' and self.mode == 'train':
            if info['train_seq'] == 0:
                option_answer = self.GPT_paraphrasing_dict[im_name]['GT_with_Rationale']
            elif info['train_seq'] == 1:
                option_answer = self.GPT_paraphrasing_dict[im_name]['GPT_paraphrasing_1']
            elif info['train_seq'] == 2:
                option_answer = self.GPT_paraphrasing_dict[im_name]['GPT_paraphrasing_2']
            elif info['train_seq'] == 3:
                option_answer = self.GPT_paraphrasing_dict[im_name]['GPT_paraphrasing_3']
            else:
                raise
        
        return im, im_path, torch.tensor(int(pid)), q_stn_out, opts, option_answer, torch.tensor(lbl), torch.tensor(answer)

    def __len__(self):
        return len(self.qa_info)

def V_COT_SMART_collate_fn(data):
    """we use it only for val and test to load the options as a list"""
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    im, im_p, pids, q_stn, opts, answer_option, lbl, answer = zip(*data)
     # im = concat(im).float()
    pids = concat(pids)
    lbl = concat(lbl)
    answer = concat(answer)
    return im, im_p, pids, q_stn, opts, answer_option, lbl, answer