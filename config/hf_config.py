"""
허깅페이스로부터 모델, 데이터, 학습 arguments를 가져오는 모듈.
"""

from functools import partial
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, List, Any
from transformers import TrainingArguments, PretrainedConfig, Seq2SeqTrainingArguments

@dataclass
class DataArguments(PretrainedConfig):
    mode='train'
    data_root="/data/SMART101-release-v1/SMART101-Data/"
    train_puzzle_list="zero_shot"
    val_puzzle_list="1,2,6,7,17,19,40,77"
    test_puzzle_list="1,2,6,7,17,19,40,77"
    train_tot=1500
    eval_tot=50

@dataclass
class ModelArguments(PretrainedConfig):
    model_type='Idefics2-8b'
    pretrained_model_path="HuggingFaceM4/idefics2-8b"
    do_image_splitting=True
    USE_LORA=True
    USE_QLORA=True
    lora_r=16
    lora_alpha=32
    lora_dropout=0.05
    max_length=20

@dataclass
class TrainingArguments(Seq2SeqTrainingArguments):
    """
        training_arguments을 상속받았기 때문에 num_train_epochs, per_device_train_batch_size등이 자동으로 들어감 
    """
    # project_name="Visual_Chain-of-Thought_Project"
    run_name="Visual_Chain-of-Thought_Project_Demo"
    num_train_epoch=3
    per_device_train_batch_size=2,
    per_device_eval_batch_size=2,
    gradient_accumulation_steps=2,
    warmup_steps=50,
    learning_rate=1e-4,
    weight_decay=0.01,
    logging_steps=25,
    output_dir="./checkpoints/dump/",
    save_strategy="steps",
    save_steps=500,
    save_total_limit=20,
    evaluation_strategy="steps"
    eval_steps=100
    load_ckpt_path=None
    bf16=True,
    remove_unused_columns=False,
    report_to='wandb',
    predict_with_generate=True
    should_log=True
    seed=1123
    ddp_find_unused_parameters=False
    
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