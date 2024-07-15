import torch
from transformers import HfArgumentParser, set_seed, Seq2SeqTrainer

import lib.V_COT_globvars as gv
from config.hf_config import *
from models.build_model import get_model
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
    set_seed(training_args.seed)
    local_rank = training_args.local_rank
    gv.custom_globals_init()
    
    # 모델 load...
    model, processor = get_model(data_args, model_args, training_args)
    
    # 데이터로더 load...
    data_module = get_dataset(training_args, model_args, data_args, processor=processor)
    
    # Trainer load...
    trainer = get_trainer(model_args, training_args, model, processor, data_module)
    
    # 테스트셋 실험...
    predictions, labels, metrics = trainer.predict(test_dataset=data_module["test_dataset"])
    # last hidden_state, hidden_state, attention_map, labels, metrics =>
    
    pred_id = predictions[0].argmax(2)
    pred_answer = processor.batch_decode(pred_id, skip_special_tokens=True)
    
    print('predictions:',predictions)
    print('labels:',labels)
    print('metrics:',metrics)
    print('\nComplete')

if __name__ == "__main__":
    eval()