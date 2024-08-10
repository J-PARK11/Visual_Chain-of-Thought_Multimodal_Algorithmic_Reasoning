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

from datasets import load_dataset
from torch.utils.data import Dataset
from transformers.image_utils import load_image
from torchvision.transforms import Compose, Normalize, Resize, ToTensor

import lib.utils as utils
import lib.V_COT_globvars as gv


class V_COT_METEOR_Dataset(Dataset):
    def __init__(self, args, mode):
        
        self.args = args
        self.mode = mode
        self.data_root = '/data/METEOR/'
        
        if self.mode == 'train':
            self.num_tot = args.train_tot        
        if self.mode != 'train':
            self.num_tot = args.eval_tot            
        
        Whole_METEOR = load_dataset("BK-Lee/Meteor")['train']
        print(f'Num of While METEOR Data: {len(Whole_METEOR)}')        
        
        # Select Available Data
        self.qa_info = []
        for i, sample in enumerate(Whole_METEOR):
            if sample['image'] != None:
                if sample['image'].split('/')[0] == '.':
                    sample['image'] = sample['image'][2:]
                img_path = os.path.join(self.data_root, sample['image'])
                if ('web' in img_path) or ('share' in img_path):
                    print(img_path)
                if os.path.exists(img_path):
                    self.qa_info.append(sample)
        print(f'Available METEOR Data: {len(self.qa_info)}')
        del Whole_METEOR
            
            
    def ans_encode(self, answer):
        return ord(answer) - ord("A")

    def __getitem__(self, idx):
        info = self.qa_info[idx]
        id, im_path = info['id'], info['image'], 
        question, rationale, answer = info['conversations'][0]['value'], info['conversations'][1]['rationale'], info['conversations'][1]['value'] 
        im = load_image(im_path)
        
        return id, im, im_path, question, rationale, answer

    def __len__(self):
        return len(self.qa_info)

class V_COT_METEOR_collator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
    
    def __call__(self, examples):
        texts = []
        images = []
        for idx, (id, im, im_path, question, rationale, answer) in enumerate(examples):
            question = "Question: " + question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": question},
                        {"type": "image"},
                        {"type": "text", "text": "Please solve the above question and explain the solution process."}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": rationale}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            
            texts.append(text.strip())
            images.append([im])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch