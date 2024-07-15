import os
import pdb
import json
import torch
import pickle
import warnings
import numpy as np
warnings.filterwarnings("ignore")

import lib.utils as utils
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import lib.V_COT_globvars as gv
from transformers.image_utils import load_image


class V_COT_reasoning_anlysis_dataset(Dataset):
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
        
        with open(self.answer_sheet_path,'r') as f:
            self.answer_sheet = json.load(f)

        # puzzle_ids = [f'{pz_id}' for pz_id in range(1,102)][:self.puzzle_tot]
        
        for puzzle_id in puzzle_ids:
            puzzle_root = puzzle_id + "/" + gv.puzzle_diff_str[self.diff] + "/"
            csv_file = "puzzle_%s%s.csv" % (puzzle_id, gv.puzzle_diff[self.diff])
            qa_info = utils.read_csv(os.path.join(self.data_root, puzzle_root, csv_file), puzzle_id)
            qa_info = qa_info[: self.num_tot]
            for t in range(len(qa_info)):
                qa_info[t]["AnswerValue"] = utils.get_val(qa_info[t], qa_info[t]["Answer"])
            self.qa_info = self.qa_info + qa_info
            
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
        
        # 정답지 있는 데이터에 한 해, 프롬프팅
        try:
            answer_sheet = self.answer_sheet[im_name]
        except:
            answer_sheet = None
        
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
        Answer_Option_phrase = ' Options:'
        for op in ["A", "B", "C", "D", "E"]:
            op_val = info[op]
            Answer_Option_phrase += f' {op}={op_val}'
            opts.append(op_val)
        q_stn = info["Question"]
        q_stn = q_stn + Answer_Option_phrase  
        
        return im, im_path, torch.tensor(int(pid)), q_stn, opts, info["Answer"], torch.tensor(lbl), torch.tensor(answer), answer_sheet

    def __len__(self):
        return len(self.qa_info)

def V_COT_SMART_collate_fn(data):
    """we use it only for val and test to load the options as a list"""
    concat = lambda data_list: torch.cat([x.unsqueeze(0) for x in data_list])
    im, im_p, pids, q_stn, opts, answer_option, lbl, answer, answer_sheet = zip(*data)
     # im = concat(im).float()
    pids = concat(pids)
    lbl = concat(lbl)
    answer = concat(answer)
    return im, im_p, pids, q_stn, opts, answer_option, lbl, answer, answer_sheet