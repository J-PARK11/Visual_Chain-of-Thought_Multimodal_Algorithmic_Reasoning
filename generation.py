
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
    print("\n========== Visual Chain of Thought Single Turn Generation ==========\n")
    
    # V_COT 결과 저장할 JSON 파일 세팅 ========================================== #
    global result_json_path
    result_json_path = os.path.join(args.save_root, args.output_name)
    # puzzle_list = args.test_puzzle_list.split(',')
    # for pids in puzzle_list:
    #     puzzle_save_root = os.path.join(args.save_root, 'puzzle', pids)
    #     if not os.path.exists(puzzle_save_root): os.makedirs(puzzle_save_root)
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
        from models.Idefics2.image_processing_idefics2 import Idefics2ImageProcessor
        
        processor = Idefics2Processor.from_pretrained("HuggingFaceM4/idefics2-8b",
                                                  do_image_splitting=True,
                                                  size={"longest_edge": 448, "shortest_edge": 378},
                                                  use_DPR=args.USE_DPR)
        # 특별한 Image Processor가 필요함.
        if args.USE_DPR:
            from models.Idefics2.modeling_DPR_idefics2 import Idefics2ForConditionalGeneration
            dpr_image_processor = Idefics2ImageProcessor(do_image_splitting=True,
                                            image_mean=[0.5,0.5,0.5], image_std=[0.5,0.5,0.5],
                                            size={"longest_edge":336, "shortest_edge":280}, # 336, 280
                                            use_DPR=True)
            processor.image_processor = dpr_image_processor
            print(f'USE DPR Processor & Model')
        else:
            from models.Idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration
                                                  
        if args.load_ckpt_path:   
            if args.USE_DPR:             
                model_path = os.path.join(args.load_ckpt_path, 'model.pth')                
                model = torch.load(model_path).to(device)
                print(f'Load DPR Model Success: {model_path}')
            else:
                model = Idefics2ForConditionalGeneration.from_pretrained(args.load_ckpt_path, 
                                                                        torch_dtype=torch.bfloat16,
                                                                        low_cpu_mem_usage=False,).to(device)
            print(f'Load ckpt: {args.load_ckpt_path}')
        else:
            model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", 
                                                                    torch_dtype=torch.bfloat16).to(device)
    print(f'\nModel Parameter numbers: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # V_COT 실행 ============================================================= #
    def Execute(epoch, args, target_dataloader):
        TP, ALL = 0, 0
        puzzle_len  = len(args.test_puzzle_list.split(','))
        for i, (im, im_path, pids, q_stn, o, ao, a, av) in tqdm(enumerate(target_dataloader)):

            # if i >= 5 : break  # 배리어
            
            im = [im]
            Whole_start_time = time.time()   
            if args.VLM_type in ['Idefics2']:
                
                exp1_pred, exp2_pred, exp3_pred, exp4_pred, exp5_pred, exp6_pred = None, None, None, None, None, None
                
                # Exp1: Q, I => A =================================================================== #
                exp1_prompt = [
                    {"role": "user",
                    "content": [
                        # {"type": "text", "text": 'Please solve the above question and return only answer like this: Answer: ?'},          # 답만 추론하는 프롬프트.
                        {"type": "text", "text": 'Looking at this image, solve the question and explain how you solved it step-by-step.'},  # Phase1 베스트 프롬프트.
                        # {"type": "text", "text": 'Please solve the above question and explain the solution process.'},                    # 학습 때와 같은 프롬프트.
                        {"type": "image"},
                        {"type": "text", "text": f'Question: {question}'}]}
                    for question in q_stn]
            
                exp1_query = processor.apply_chat_template(exp1_prompt, add_generation_prompt=True)
                exp1_query_input = processor(text=exp1_query, images=im, return_tensors='pt').to(device)
                with torch.no_grad():
                    exp1_pred_id = model.generate(**exp1_query_input, max_new_tokens=600)
                    exp1_pred = processor.batch_decode(exp1_pred_id, skip_special_tokens=True)
                
                for j in range(len(exp1_pred)):
                    exp1_pred[j] = exp1_pred[j].split('Assistant: ')[-1]
                    print(q_stn[j])
                    print(exp1_pred[j])
                    print(ao[j][-1])
                    
            # Result Logging                                
            for iter, img_path in enumerate(im_path):
                
                st = exp1_pred[iter].upper().find('ANSWER: ')
                if st<0:
                    hit = False
                else:
                    ed = st + 9
                    pred_answer = exp1_pred[iter][st:ed][-1]
                    hit = (pred_answer == ao[iter][-1])
                    if hit : TP+=1
                    ALL += 1
                
                log_dict=dict()
                img_name = img_path.split('/')[-1]                    
                question = q_stn[iter]
                result_json[img_name] = {'Question': question,
                                         'Only_answer':exp1_pred[iter],
                                        #  'Answer_with_Reasoning':exp2_pred[iter],
                                        #  'Image_Caption':exp6_pred[iter],
                                         'GT_option': ao[iter][-1],
                                         'GT_value': o[iter][option_dict[ao[iter][-1]]],
                                         'Hit': hit}
                
            if i % 100 == 0:
                with open(result_json_path,'w') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=4)
                
                # print('\n', img_name, exp2_pred[iter])
                # puzzle_save_path = os.path.join(args.save_root, 'puzzle', str(int(pids)), img_name)
                # plt.figure(figsize=(6,6))
                # plt.suptitle(f'V-COT Reasoning: "{img_name}"')
                # plt.imshow(im[iter][0])
                # plt.axis('off')
                # plt.savefig(puzzle_save_path)
                # plt.clf()
                
                """
                print(img_name)
                print(q_stn[0])
                print(ao[0])
                print(o[0][option_dict[ao[iter][-1]]])
                print(exp1_pred[0])
                """

                # ================================================================ #
                    
            Whole_end_time = time.time()
            Whole_dur_time = Whole_end_time - Whole_start_time
            print(f' Batch: {i}/{len(target_dataloader)}  Dur_time: {Whole_dur_time:.4f} for {len(im)} images')
            try: print(f"Accuracy = {TP}/{ALL} = {TP/ALL:.4f}")
            except: pass

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
    parser.add_argument("--load_ckpt_path", type=str, default=None)
    parser.add_argument("--output_name", type=str, default="dump.json")
    parser.add_argument("--task", type=str, default=None)
    parser.add_argument("--test_puzzle_list", type=str, default='1,2,6,7,17,19,40,77')
    parser.add_argument("--eval_tot", type=int, default=3)
    
    # 내가 추가한 Argument List =================================================================== #
    parser.add_argument("--VLM_type", type=str, default='Idefics2')
    parser.add_argument("--gpu_num", type=int, default=0, help="Define GPU used")
    parser.add_argument("--USE_DPR", type=bool, default=False)
    
    # 세팅
    args = parser.parse_args()
    gv.custom_globals_init()  
    dataloader = get_data_loader(args, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # V-COT 시작
    global device
    device = f'cuda:{args.gpu_num}'
    V_COT(args, dataloader)