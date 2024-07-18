import os
import pdb
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

# import datasets_lib.load_iconqa as load_iconqa #강현 추가
# import datasets_lib.load_mathverse as load_mathverse #강현 추가
# import datasets_lib.load_mathvision as load_mathvision #강현 추가
# import datasets_lib.load_mathvista as load_mathvista #강현 추가
# import datasets_lib.load_mmbench as load_mmbench #강현 추가
# import datasets_lib.load_mmmu as load_mmmu #강현 추가
# import datasets_lib.load_mmstar as load_mmstar #강현 추가
# import datasets_lib.load_scienceqa as load_scienceqa #강현 추가


class V_COT_SMART101_Dataset(Dataset):
    def __init__(self, args, mode):
        # super().__init__(args)
        self.data_root = args.data_root
        self.mode = mode
        self.answer_sheet_path = './V_COT_output/V_COT_Answer_Sheet.json'
        
        if self.mode == 'train':
            if args.train_puzzle_list == 'all':
                puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
            elif args.train_puzzle_list == 'zero_shot':
                puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)]
                puzzle_ids = sorted(list(set(puzzle_ids) - set(['1','2','6','7','17','19','40','77'])))
            else:
                self.puzzle_list = args.train_puzzle_list
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
        
        self.diff = 'easy'
        self.qa_info = []
        self.args = args
        self.add_data = args.add_data
        
        with open(self.answer_sheet_path,'r') as f:
            self.answer_sheet = json.load(f)
        
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff]
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
                qa_info[t]["image"] = self.data_root + qa_info[t]["puzzle_id"]+ "/img/" + qa_info[t]["image"]
            self.qa_info = self.qa_info + qa_info
        
        if self.add_data != None and self.mode == 'train':
            if "iconqa" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_iconqa'), f'generate_iconqa_qainfo')('/data/SMART101/iconqa_data') + self.qa_info
            if "mathverse" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mathverse'), f'generate_mathverse_qainfo')('/data/SMART101/MathVerse/testmini.json') + self.qa_info
            if "mathvision" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mathvision'), f'generate_mathvision_qainfo')("MathLLMs/MathVision") + self.qa_info
            if "mathvista" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mathvista'), f'generate_mathvista_qainfo')("AI4Math/MathVista") + self.qa_info
            if "mmbench" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mmbench'), f'generate_mmbench_qainfo')("HuggingFaceM4/MMBench_dev") + self.qa_info
            if "mmmu" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mmmu'), f'generate_mmmu_qainfo')("MMMU/MMMU") + self.qa_info
            if "mmstar" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_mmstar'), f'generate_mmstar_qainfo')('/data/SMART101/MMStar') + self.qa_info
            if "scienceqa" in self.add_data:
                self.qa_info = getattr(globals().get(f'load_scienceqa'), f'generate_scienceqa_qainfo')('/data/SMART101/ScienceQA') + self.qa_info
            
        if self.mode == 'train':
            random.seed(1123)
            random.shuffle(self.qa_info)
            print('Train Dataset shuffled')
            
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        pid = info["puzzle_id"]
        puzzle_root = pid + "/" + gv.puzzle_diff_str[self.diff] + "/"
        
        im_path = info["image"] # 강현 추가
        im_name = im_path.split('/')[-1]
        
        if im_path=='str': #강현 추가
            im = Image.open(im_path) # 강현 추가
        else: #강현 추가
            im = im_path #강현 추가
        lbl = self.ans_encode(info["Answer"])
        answer_value = info["AnswerValue"]
        answer = ''
        
        # 정답지 있는 데이터에 한 해, 프롬프팅
        try:
            answer_sheet = self.answer_sheet[im_name]
        except:
            answer_sheet = None
        
        # 시퀀스 데이터 예외처리.
        """
        if int(pid) not in gv.SEQ_PUZZLES:
            answer[0] = answer_value
        else:
            try:
                answer[: len(answer_value)] = answer_value
            except:
                print(info)
                pdb.set_trace()
        """
        
        if not info: info = []
        opts = []
        Answer_Option_phrase = '\nOptions:'
        op_list = [key for key in ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H'] if key in info] #강현 추가
        for op in op_list:
            op_val = info[op]
            Answer_Option_phrase += f' {op}={op_val}'
            opts.append(op_val)
        q_stn = info["Question"]
        q_stn = q_stn + Answer_Option_phrase  
        
        return im, im_path, torch.tensor(int(pid)), q_stn, opts, info["Answer"], torch.tensor(lbl), answer, answer_sheet #강현 추가(수정함 answer를 str로도 받을 수 있게 끔)

    def __len__(self):
        return len(self.qa_info)

def V_COT_SMART_collate_fn(data): #강현 추가(수정함 answer를 str로도 받을 수 있게 끔)
    """we use it only for val and test to load the options as a list"""
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    im, im_p, pids, q_stn, opts, answer_option, lbl, answer, answer_sheet = zip(*data) 
     # im = concat(im).float()
    pids = concat(pids)
    lbl = concat(lbl)
    # answer = concat(answer)
    return im, im_p, pids, q_stn, opts, answer_option, lbl, answer, answer_sheet