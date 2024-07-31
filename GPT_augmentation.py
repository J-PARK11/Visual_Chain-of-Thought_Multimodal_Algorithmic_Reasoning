import os
import json
import torch
import base64
import requests
import argparse
from PIL import Image
from tqdm import tqdm
from openai import OpenAI
import lib.V_COT_globvars as gv
import datasets_lib.V_COT_data_loader as dl

os.environ["OPENAI_API_KEY"] = "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"

def get_data_loader(args, batch_size=100, shuffle=False, num_workers=6, pin_memory=True):
    args.preprocess = None
    dataset = dl.V_COT_SMART101_Dataset(args, mode='train')
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

def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode("utf-8")

def augmentation(args, target_dataloader):
    client = OpenAI()    
    new_json = dict()
    option_dict = {0:'A', 1:'B', 2:'C', 3:'D', 4:'E'}     
    prompt = "Here is the solution process and answer for a example puzzle similar to the ones you will solve. Please refer to this answer to transform the solution process for the upcoming puzzle and question. dont include question it self."
    
    puzzle_len  = int(len(target_dataloader) / args.train_tot)
    print("\n========== GPT Instance Puzzle Augmentation ==========")
    print(f'Batch Size: {args.batch_size}, #puzzle: {puzzle_len}, #instance: {args.train_tot}, #Data: {len(target_dataloader)}\n')
    for i, (im, im_path, pids, q_stn, o, ao, a, av) in tqdm(enumerate(target_dataloader)):
        img_name = im_path[0].split('/')[-1]    
        im = encode_image(im_path[0]) #불러온 csv에서 이미지의 이름을 찾고 이미지 경로에서 이미지 불러오기                
        response = client.chat.completions.create(
            model="gpt-4o",
            messages=[
                {"role": "user", "content": [
                    {"type": "text", "text": prompt },
                    {"type": "text", "text": f"Example answer: {ao[0]}" },
                    {"type": "image_url", "image_url": {
                        "url": f"data:image/png;base64,{im}"}
                    },
                    {"type": "text", "text": f"Question: {q_stn[0]}. Answer: {option_dict[int(a)]}={o[0][int(a)]}" }
                ]}
            ],
        temperature=0.5)
        new_json[img_name] = dict()
        new_json[img_name][f'GT_with_Rationale'] = response.choices[0].message.content
        print(f'{img_name}, loop_index: {i}')
        # print(img_name, response.choices[0].message.content)
    
    # Save the JSON data to a file
    with open(args.output_name, "w") as json_file:
        json.dump(new_json, json_file, indent=4)  # Use indentation for better readability

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
    parser.add_argument("--input_name", type=str, default="dump.json")
    parser.add_argument("--output_name", type=str, default="./V_COT_output/GT/GT_rationale_dataset_develop.json")
    parser.add_argument("--train_puzzle_list", type=str, default='GPT_augmentation')
    parser.add_argument("--train_tot", type=int, default=1000)
    
    # 내가 추가한 Argument List =================================================================== #
    parser.add_argument("--VLM_type", type=str, default='Idefics2')
    
    # 세팅
    args = parser.parse_args()
    gv.custom_globals_init()  
    dataloader = get_data_loader(args, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers)
    
    # 증강 시작
    augmentation(args, dataloader)
    print("\n========== Complete ==========")