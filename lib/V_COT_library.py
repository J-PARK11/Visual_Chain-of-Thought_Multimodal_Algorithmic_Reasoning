import os
import csv
from PIL import Image
import matplotlib.pyplot as plt
from matplotlib.image import imread

# {pid:insid}
def get_insid(pid):
    matched_insid = {"7":"2", "10":"1"}
    insid = matched_insid[pid]
    return pid, insid

# Function
def read_csv(csvfilename, puzzle_id):

    qa_info = []
    option = ["A", "B", "C", "D", "E"]
    with open(csvfilename, newline="") as csvfile:
        datareader = csv.DictReader(csvfile)
        for row in datareader:
            
            row["puzzle_id"] = str(puzzle_id)
            if len(row["A"]) == 0:
                row["A"] = "A"
                row["B"] = "B"
                row["C"] = "C"
                row["D"] = "D"
                row["E"] = "E"
            qa_info.append(row)
            
            # create an option column
            row["Option"] = "\n".join([f"{o}. {row[o]}" for o in option])
    return qa_info

def img_show(img_path):
    img = imread(img_path)
    plt.imshow(img)
    plt.grid(False)
    plt.axis('off')
    plt.show()
    
def get_v_cot_img_and_question(pid, img_root_path, question, option):
    q_dict={}
    
    if pid == '7':
        q_dict['Q1'] = f"Please return the important information in the image with a brief explanation."
        q_dict['Q2'] = f"How many mailboxes does one ball trade for?"
        q_dict['Q3'] = f"So, how many umbrellas does one mailbox trade for?"
        # q_dict['Q4'] = f"How many umbrellas does two balls trade for based on earlier trade relationship?\nAnswer Options:\n{option} Please answer with a commentary."
        q_dict['Q4'] = f"Refer to the previous conversation and answer the following questions.\n{question}\nAnswer Options:\n{option}\nPlease answer with a commentary."

        q_dict['I1'] = Image.open(os.path.join(img_root_path, "I1.png"))
        q_dict['I2'] = Image.open(os.path.join(img_root_path, "I2.png"))
        q_dict['I3'] = Image.open(os.path.join(img_root_path, "I3.png"))
        q_dict['I4'] = Image.open(os.path.join(img_root_path, "I1.png"))
    
    elif pid == '10':
        q_dict['Q1'] = f"Please return the important information in the image with a brief explanation."
        q_dict['Q2'] = f"Use the symbol at the top-left to make an equation to match the number at the bottom-right."
        q_dict['Q3'] = f"Similarly, to match the numbers at the bottom-right, use the symbols at the top-left to calculate the numbers that will fit in the black square at the top-right."
        q_dict['Q4'] = f"Refer to the previous conversation and answer the following questions.\n{question}\nAnswer Options:\n{option}\nPlease answer with a commentary."

        q_dict['I1'] = Image.open(os.path.join(img_root_path, "I1.png"))
        q_dict['I2'] = Image.open(os.path.join(img_root_path, "I2.png"))
        q_dict['I3'] = Image.open(os.path.join(img_root_path, "I3.png"))
        q_dict['I4'] = Image.open(os.path.join(img_root_path, "I4.png"))
    
    
    return q_dict

# 프롬프트
def get_prompt_format(question_dict):
    first_prompt = [
        
        # First-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q1']}"},
                ]},
    ]

    second_prompt = [
        
        # First-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q1']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Second-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q2']}"},
                ]},
    ]

    third_prompt = [
        
        # First-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q1']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Second-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q2']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Third-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q3']}"},
                ]},
    ]

    fourth_prompt = [
        
        # First-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q1']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Second-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q2']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Third-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q3']}"},
                ]},
        {"role": "assistant",
        "content": [
            {"type": "text", "text": ''}
        ]},
        
        # Fourth-Turn
        {"role": "user",
        "content": [
                {"type": "image"},
                {"type": "text", "text": f"{question_dict['Q4']}"}, 
                ]},
    ]
    return first_prompt, second_prompt, third_prompt, fourth_prompt