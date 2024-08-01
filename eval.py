import os
import wandb
import torch
from transformers import HfArgumentParser, set_seed

import lib.V_COT_globvars as gv
from config.hf_config import *
from models.build_model import get_model
from metrics.build_metric import get_metric
from trainers.build_trainer import get_trainer
from datasets_lib.build_dataset import get_dataset

import warnings
warnings.filterwarnings('ignore')

def eval():
    global local_rank
    
    # 허깅페이스 argument parser load...
    parser = HfArgumentParser(
            (DataArguments, ModelArguments, TrainingArguments))
    data_args, model_args, training_args = parser.parse_args_into_dataclasses()
    print(f'\n{data_args}\n{model_args}\n{training_args}\n')
    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    gv.custom_globals_init()
    
    if training_args.report_to == ['wandb']:
        os.environ["WANDB_PROJECT"] = training_args.run_name
        wandb.init(project=training_args.run_name)
        wandb.run_name = training_args.run_name
    
    # 모델 load...
    model, processor = get_model('train', data_args, model_args, training_args)
    
    # 데이터로더 load...
    data_module = get_dataset(training_args, model_args, data_args, processor=processor)
    
    # Trainer load...
    metric = get_metric(model_args, data_args, processor, info='base')
    trainer = get_trainer(model_args, training_args, model, processor, data_module, metric)
    
    # 테스트셋 실험...
    predictions, labels, metrics = trainer.predict(test_dataset=data_module["test_dataset"])
    # last hidden_state, hidden_state, attention_map, labels, metrics =>
    
    pred_id = predictions[0].argmax(2)
    pred_answer = processor.batch_decode(pred_id, skip_special_tokens=True)
    print(pred_answer[0])
    
    # print('predictions:',predictions)
    # print('labels:',labels)
    # print('metrics:',metrics)
    print('\nComplete')

if __name__ == "__main__":
    eval()