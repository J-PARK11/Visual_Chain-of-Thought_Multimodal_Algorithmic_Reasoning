"""
Visual Chain-of-Thought Reasoning Analysis

: Total Experiment Setting 5
    1. Q, I => A
    2. Q, I => A, Text Reasoning(TR)
    3. Q, I => A, Visual Patch(VP)
    4. Q, I, TR => A
    5. Q, I, VP => A

    - VLM_type: ['IBLIP', 'Idefics2']
"""

# python V_COT_reasoning_analysis.py --num_workers 8 --batch_size 4 --data_root ./dataset/test-images/ --save_root ./checkpoints/dump --gpu_num 0

import os
import copy
import json
import time
import torch
import warnings
import argparse
import numpy as np
from PIL import Image
from tqdm import tqdm
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch.utils.data.distributed import DistributedSampler

warnings.filterwarnings("ignore")
os.environ["TOKENIZERS_PARALLELISM"] = "1"
global option_dict
option_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}

import lib.V_COT_globvars as gv
import datasets_lib.V_COT_data_loader as dl

def V_COT(args, dataloader):
    print("\n========== Visual Chain of Thought Reasoning Analysis Start ==========\n")
    
    # V_COT 결과 저장할 JSON 파일 세팅 ========================================== #
    global result_json_path
    result_json_path = os.path.join(args.save_root, args.output_name)
    puzzle_list = args.test_puzzle_list.split(',')
    for pids in puzzle_list:
        puzzle_save_root = os.path.join(args.save_root, 'puzzle', pids)
        if not os.path.exists(puzzle_save_root): os.makedirs(puzzle_save_root)
    print(f'Result Json path: {result_json_path}')
            
    global result_json
    result_json = dict()
    # ======================================================================= #
    
    # VLM 모델 정의 =========================================================== #
    global model
    global processor
    if args.VLM_type == "IBLIP":
        from transformers import InstructBlipProcessor, InstructBlipForConditionalGeneration
        model = InstructBlipForConditionalGeneration.from_pretrained("Salesforce/instructblip-flan-t5-xl").to(torch.bfloat16).to(device)
        processor = InstructBlipProcessor.from_pretrained("Salesforce/instructblip-flan-t5-xl")
        
    elif args.VLM_type == "Idefics2":
        from models.Idefics2.processing_idefics2 import Idefics2Processor
        from models.Idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration
        processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b",
                                                  do_image_splitting=False)
                                                #   size= {"longest_edge": 448, "shortest_edge": 378})
        if args.load_ckpt_path:                                                
            model = Idefics2ForConditionalGeneration.from_pretrained(args.load_ckpt_path, 
                                                                    torch_dtype=torch.bfloat16).to(device)
        else:
            model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", 
                                                                    torch_dtype=torch.bfloat16).to(device)

    # V_COT 실행 ============================================================= #
    def Execute(epoch, args, target_dataloader):
        puzzle_len  = len(args.test_puzzle_list.split(','))
        print(f'Batch Size: {args.batch_size}, #puzzle: {puzzle_len}, #instance: {args.eval_tot}, #Data: {len(target_dataloader)}\n')
        for i, (im, im_path, pids, q_stn, o, ao, a, av, answer_sheet) in tqdm(enumerate(target_dataloader)):

            # if i >= 5 : break  # 배리어
            
            im = [im]
            Whole_start_time = time.time()   
            if args.VLM_type in ['Idefics2']:
                
                # Exp1: Q, I => A =================================================================== #
                exp1_prompt = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": 'Looking at this image, solve the question.'},
                         {"type": "image"},
                         {"type": "text", "text": f'Question: {question}'}]}
                    for question in q_stn]
                
                exp1_query = processor.apply_chat_template(exp1_prompt, add_generation_prompt=True)
                exp1_query_input = processor(text=exp1_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp1_pred_id = model.generate(**exp1_query_input, max_new_tokens=50)
                    exp1_pred = processor.batch_decode(exp1_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp1_pred)):
                    exp1_pred[j] = exp1_pred[j].split('Assistant: ')[-1]
                    
                # Exp2: Q, I => A, Solving Process ================================================== #
                exp2_prompt = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": 'Looking at this image, solve the question and explain how you solved it step-by-step.'},
                         {"type": "image"},
                         {"type": "text", "text": f'Question: {question}'}]}
                    for question in q_stn]
                
                exp2_query = processor.apply_chat_template(exp2_prompt, add_generation_prompt=True)
                exp2_query_input = processor(text=exp2_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp2_pred_id = model.generate(**exp2_query_input, max_new_tokens=500)
                    exp2_pred = processor.batch_decode(exp2_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp2_pred)):
                    exp2_pred[j] = exp2_pred[j].split('Assistant: ')[-1]
                
                # Exp3: Q, I => A, Core Visual Patch Coordinates ==================================== #
                exp3_prompt = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": 'Looking at this image, solve the question and return the bounding box coordinates that are the key to solving the problem.'},
                         {"type": "image"},
                         {"type": "text", "text": f'Question: {question}'}]}
                    for question in q_stn]
                
                exp3_query = processor.apply_chat_template(exp3_prompt, add_generation_prompt=True)
                exp3_query_input = processor(text=exp3_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp3_pred_id = model.generate(**exp3_query_input, max_new_tokens=500)
                    exp3_pred = processor.batch_decode(exp3_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp3_pred)):
                    exp3_pred[j] = exp3_pred[j].split('Assistant: ')[-1]
                    
                # Exp4: Q, I, GT Reasoning step => A ================================================ #
                exp4_prompt = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": 'Looking at this image and the solving process, answer the question and explain the solution process.'},
                         {"type": "image"},
                         {"type": "text", "text": f'Solving process: {reasoning["Reasoning_Step"]}\nQuestion: {question}'}]}
                    for question, reasoning in zip(q_stn, answer_sheet)]
                
                exp4_query = processor.apply_chat_template(exp4_prompt, add_generation_prompt=True)
                exp4_query_input = processor(text=exp4_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp4_pred_id = model.generate(**exp4_query_input, max_new_tokens=500)
                    exp4_pred = processor.batch_decode(exp4_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp4_pred)):
                    exp4_pred[j] = exp4_pred[j].split('Assistant: ')[-1]                  
                
                # Exp5: Q, I, GT Answer sheet => A ================================================== #
                exp5_prompt = [
                    {"role": "user",
                     "content": [
                         {"type": "text", "text": 'Looking at this image and the solving process, answer the question and explain the solution process.'},
                         {"type": "image"},
                         {"type": "text", "text": f'Solving process: {solution["Answer_Sheet"]}\nQuestion: {question}'}]}
                    for question, solution in zip(q_stn, answer_sheet)]
                
                exp5_query = processor.apply_chat_template(exp5_prompt, add_generation_prompt=True)
                exp5_query_input = processor(text=exp5_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp5_pred_id = model.generate(**exp5_query_input, max_new_tokens=500)
                    exp5_pred = processor.batch_decode(exp5_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp5_pred)):
                    exp5_pred[j] = exp5_pred[j].split('Assistant: ')[-1]  
                
            # Result Logging                                 
            for iter, img_path in enumerate(im_path):
                log_dict=dict()
                img_name = img_path.split('/')[-1]                    
                question = q_stn[iter]
                result_json[img_name] = {'Question': question,
                                        #  'Option': o[iter].tolist(),
                                         'exp1':exp1_pred[iter],
                                         'exp2':exp2_pred[iter],
                                         'exp3':exp3_pred[iter],
                                         'exp4':exp4_pred[iter],
                                         'exp5':exp5_pred[iter],
                                         'GT_option': ao[iter],
                                         'GT_value': o[iter][option_dict[ao[iter]]]}
                
                puzzle_save_path = os.path.join(args.save_root, 'puzzle', str(int(pids)), img_name)
                plt.figure(figsize=(6,6))
                plt.suptitle(f'V-COT Reasoning: "{img_name}"')
                plt.imshow(im[iter][0])
                plt.axis('off')
                plt.savefig(puzzle_save_path)
                plt.clf()
                    
                # ================================================================ #
                    
            Whole_end_time = time.time()
            Whole_dur_time = Whole_end_time - Whole_start_time
            print(f' Batch: {i}/{len(target_dataloader)}  Dur_time: {Whole_dur_time:.4f} for {len(im)} images')

    for epoch in range(1):

        # V-COT 실행
        Execute(epoch, args, dataloader)
                
        # JSON 결과 파일 저장 및 확인
        with open(result_json_path,'w') as f:
            json.dump(result_json, f, ensure_ascii=False, indent=4)
        
        with open(result_json_path,'r') as f:
            saved_caption = json.load(f)
            print(f'Saved Result: {len(saved_caption)}')
        
    print('\n================= Complete =================')


def get_data_loader(args, batch_size=100, shuffle=False, num_workers=6, pin_memory=True):
    args.preprocess = None
    dataset = dl.V_COT_SMART101_Dataset(args, mode='test')
    collate_fn = dl.V_COT_SMART_collate_fn

    data_loader = torch.utils.data.DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=pin_memory,
        collate_fn=collate_fn,
        )
        
    return data_loader

if __name__ == "__main__":
    
    parser = argparse.ArgumentParser(description="SMART dataset")
    parser.add_argument("--batch_size", default=1, type=int)
    parser.add_argument("--num_workers", default=8, type=int)
    parser.add_argument(
        "--data_root",
        type=str,
        default="/data/SMART101-release-v1/SMART101-Data/",
        help="location of the csv files, and location of the images, relative location is provided in the csv file.",
    )
    parser.add_argument("--save_root", type=str, default="./V_COT_output/", help="location to save intermediate files.")
    parser.add_argument("--load_ckpt_path", type=str, default="./checkpoints/dump/")
    parser.add_argument("--output_name", type=str, default="dump.json")
    parser.add_argument("--test_puzzle_list", type=str, default='1,2,6,7,17,19,40,77')
    parser.add_argument("--eval_tot", type=int, default=3)
    
    # 내가 추가한 Argument List =================================================================== #
    parser.add_argument("--VLM_type", type=str, default='Idefics2')
    parser.add_argument("--gpu_num", type=int, default=0, help="Define GPU used")
    
    
    # 세팅
    args = parser.parse_args()
    gv.custom_globals_init()  
    dataloader = get_data_loader(args, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # V-COT 시작
    global device
    device = f'cuda:{args.gpu_num}'
    V_COT(args, dataloader)