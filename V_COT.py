
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
global need_q_pid
option_dict = {'A':0, 'B':1, 'C':2, 'D':3, 'E':4}
need_q_pid = ['1', '2', '6', '7', '9', '11', '12', '14', '20', '21', '22', '25', '26', '27', '29', '30', '31', '32', '35', '36', '38', '39', '43', '44', '46', '47', '48', '49', '50', '52', '53', '54', '55', '57', '59', '60', '62', '63', '64', '65', '66', '67', '68', '69', '70', '71', '72', '73', '74', '75', '76', '81', '84', '86', '87', '88', '89', '90', '91', '92', '93', '96', '98', '100', '101']

import lib.V_COT_globvars as gv
import datasets_lib.V_COT_data_loader as dl

def V_COT(args, dataloader):
    print("\n========== Visual Chain of Thought Multi Turn Generation ==========\n")
    
    # V_COT 결과 저장할 JSON 파일 세팅 ========================================== #
    global result_json_path
    global inter_reason_q_dict
    result_json_path = os.path.join(args.save_root, args.output_name)
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
                                            size={"longest_edge":336, "shortest_edge":280}, # 336, 280 / 224, 190
                                            use_DPR=args.USE_DPR)
            processor.image_processor = dpr_image_processor
        else:
            from models.Idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration
                                                  
        if args.load_ckpt_path:                                                
            model = Idefics2ForConditionalGeneration.from_pretrained(args.load_ckpt_path, 
                                                                    torch_dtype=torch.bfloat16).to(device)
            print(f'Load ckpt: {args.load_ckpt_path}')
        else:
            model = Idefics2ForConditionalGeneration.from_pretrained("HuggingFaceM4/idefics2-8b", 
                                                                    torch_dtype=torch.bfloat16).to(device)
    print(f'\nModel Parameter numbers: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')

    # V_COT 실행 ============================================================= #
    def Execute(epoch, args, target_dataloader):
        
        with open(args.inter_reason_q_path,'r') as f:
            inter_reason_q_dict = json.load(f)
            print(f'Inter Mediate Question: {len(inter_reason_q_dict)}')
        
        TP, ALL = 0, 0
        puzzle_len  = len(args.test_puzzle_list.split(','))
        for i, (im, im_path, pids, q_stn, o, ao, a, av) in tqdm(enumerate(target_dataloader)):

            # if i >= 5 : break  # 배리어
            
            im = [im]
            Whole_start_time = time.time()   
            if args.VLM_type in ['Idefics2']:
                
                pid = str(int(pids))
                
                # if pid in need_q_pid: continue
                
                prompt_bundle = inter_reason_q_dict[pid]
                prompt_bundle = list(prompt_bundle.values())
                num_turn = len(prompt_bundle)
                need_q_in_first = (pid in need_q_pid)
                query_bundle = get_query_format_from_prompt(prompt_bundle, num_turn, q_stn, need_q_in_first)
                                
                prev_turn_answer = []
                for q_i, query in enumerate(query_bundle):
                    
                    # Multi-turn Query Transfer
                    for q_t in range(q_i):
                        update_idx = 2*(q_t+1)-1
                        query[update_idx]['content'][0]['text'] = prev_turn_answer[q_t]
                        
                    processed_query = processor.apply_chat_template(query, add_generation_prompt=True)
                    processed_input = processor(text=processed_query, images=im, return_tensors="pt").to(device)
                    with torch.no_grad():    
                        pred_id = model.generate(**processed_input, max_new_tokens=600)
                        pred = processor.batch_decode(pred_id, skip_special_tokens=True)
                        answer = pred[0].split('\nAssistant: ')[-1].strip().replace('\n\n', '\n').replace('\\', '').replace('   ', ' ')
                        prev_turn_answer.append(answer)
                        
                query.append({'role':'assistant', 'Answer':f'{answer}'})                        
                print(answer)
                
                st = answer.upper().find('ANSWER: ')
                if st<0:
                    hit = False
                else:
                    ed = st + 9
                    pred_answer = answer[st:ed][-1]
                    hit = (pred_answer == ao[0][-1])
                    if hit : TP+=1
                    ALL += 1
                
                log_dict=dict()
                img_name = im_path[0].split('/')[-1]                    
                question = q_stn[0]
                result_json[img_name] = {'Question': question,
                                         'Turn1': prev_turn_answer[0],
                                         'Turn2': prev_turn_answer[1],
                                         'Turn3': prev_turn_answer[2],
                                         'Turn4': prev_turn_answer[3],
                                         'GT_option': ao[0][-1],
                                         'GT_value': o[0][option_dict[ao[0][-1]]],
                                         'Hit': hit}
            
            if i < 100 == 0:
                with open(result_json_path,'w') as f:
                    json.dump(result_json, f, ensure_ascii=False, indent=4)
                                                 
                # ============= =================================================== #
                    
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

def get_query_format_from_prompt(prompt_bundle, num_turn, q_stn, need_q_in_first):
    query_bundle = []
    
    if need_q_in_first:
        prompt_bundle[0] = f"Question: {q_stn[0]} {prompt_bundle[0]}"
    
    for n in range(num_turn):
        query = []
        for i in range(n):
            iterative_role_sentence = {"role": "user",
                                        "content": [
                                                {"type": "text", "text": f"{prompt_bundle[i]}"},
                                                ]}
            iterative_assistant_sentence = {"role": "assistant",
                                            "content": [
                                                {"type": "text", "text": ''}
                                            ]}
            query.append(iterative_role_sentence), query.append(iterative_assistant_sentence)
        
        main_senetence = {"role": "user",
                            "content": [
                                    {"type": "image"},
                                    {"type": "text", "text": f"{prompt_bundle[n]}"},
                                    ]}
        query.append(main_senetence)
        query_bundle.append(query)
    
    return query_bundle

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
    parser.add_argument("--inter_reason_q_path", type=str, default="./V_COT_output/GT/intermediate_reasoning_question.json")
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
    args.USE_DPR = False
    print(args)
    V_COT(args, dataloader)