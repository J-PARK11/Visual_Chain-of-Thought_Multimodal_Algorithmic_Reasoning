"""
허깅페이스로부터 모델, 데이터, 학습 arguments를 가져오는 모듈.
"""

from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any
from transformers import TrainingArguments, PretrainedConfig, Seq2SeqTrainingArguments

@dataclass
class DataArguments():
    task: str=field(default="supervised")
    data_root: str=field(default="/data/SMART101-release-v1/SMART101-Data/")
    train_puzzle_list: str=field(default=None)
    val_puzzle_list: str=field(default="1,2,6,7,10,17,19,40,77,80,81,82,83,85,88,92,95")
    test_puzzle_list: str=field(default="1,2,6,7,10,17,19,40,77,80,81,82,83,85,88,92,95")
    train_tot: int=field(default=1000)
    eval_tot: int=field(default=3)
    add_data: str=field(default=None)
    gpt_data_include_level: int=field(default=2)
    USE_DPR: int=field(default=None)
    GT_with_rationale_dict_path: str=field(default='./V_COT_output/GT/GT_rationale_dataset_develop.json')
    GPT_paraphrasing_dict_path: str=field(default='./V_COT_output/GT/gpt_paraphrasing_result.json')
    GPT_augmentation_dict_path: str=field(default='./V_COT_output/GPT_aug/GPT_augmented_101000/gpt_augmentation_result_total.json')
    
    # {'custom', 'supervised', 'zero_shot', 'GT_with_rationale', 'GPT_augmentation_generation', 'GPT_augmentation_train'}
    # 1, 2, 6, 7, 19, 40, 44, 77
    # 50, 51, 54, 55, 56, 58, 61, 78
    # 80, 81, 82, 83, 85, 88, 92, 95

@dataclass
class ModelArguments(PretrainedConfig):
    model_type: str=field(default='Idefics2-8b')
    pretrained_model_path: str=field(default="HuggingFaceM4/idefics2-8b")
    do_image_splitting: bool=field(default=True)
    USE_LORA: bool=field(default=True)
    USE_QLORA: bool=field(default=True)
    lora_r: int=field(default=16)
    lora_alpha: int=field(default=32)
    lora_dropout: float=field(default=0.05)
    max_length: int=field(default=20)
    return_dict: bool=field(default=True)
    load_ckpt_path: str=field(default=None) # "./checkpoints/GPT_augmentation/checkpoint-1344/"

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):

    # project_name="Visual_Chain-of-Thought_Project"
    run_name: str=field(default="Visual_Chain-of-Thought_Project_Demo")
    num_train_epoch: int=field(default=3)
    per_device_train_batch_size: int=field(default=2)
    per_device_eval_batch_size: int=field(default=2)
    gradient_accumulation_steps: int=field(default=2)
    warmup_steps: int=field(default=50)
    learning_rate: float=field(default=1e-4)
    weight_decay: float=field(default=0.01)
    logging_steps: int=field(default=25)
    output_dir: str=field(default="./checkpoints/dump/")
    save_strategy: str=field(default="steps")
    save_steps: int=field(default=500)
    save_total_limit: int=field(default=20)
    evaluation_strategy: str=field(default="steps")
    eval_steps: int=field(default=100)
    bf16: bool=field(default=True)
    remove_unused_columns: bool=field(default=False)
    report_to: str=field(default='wandb')
    predict_with_generate: bool=field(default=True)
    should_log: bool=field(default=True)
    seed: int=field(default=1123)
    ddp_find_unused_parameters: bool=field(default=False)
    
    # push_to_hub_model_id="idefics2-8b-docvqa-finetuned-tutorial",
    # evaluation_strategy: str="epoch",
    
    # optim: str = field(default="adamw_torch")
    # label_names: List[str]=field(default_factory=partial(list, ["labels"]))
    # seed: int=42
    # should_log: bool=True
    # pretrained_module_lr: float=field(default=1e-6) #learning rate for pretrained moduel
    # scratch_module_lr: float=field(default=1e-4) #learning rate for modules which are trained from scratch
    # predict_with_generate: bool=True # evaluate시 AR방식으로 생성해서 결과 뽑아주게함. False면 teacher forcing
    # max_length=256