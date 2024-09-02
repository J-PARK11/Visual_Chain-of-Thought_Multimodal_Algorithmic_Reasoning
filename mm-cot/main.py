import os
import numpy as np
import torch
import os
import re
import wandb
import json
import argparse
import random

import warnings
warnings.filterwarnings(action='ignore')

from transformers import AutoTokenizer, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, T5ForConditionalGeneration
from model import T5ForMultimodalGeneration
from utils_data import img_shape, MMCOT_SMART_Dataset, ScienceQADatasetImg, load_img_features
from utils_prompt import *
from rich.table import Column, Table
from rich import box
from rich.console import Console
console = Console(record=True)
import nltk
nltk.download('punkt')
import evaluate
import lib.globvars_smart as gv

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--proj_name', type=str, default='mmcot_rationale_generation_project')
    parser.add_argument('--run_name', type=str, default='mmcot_rationale_generation_run')
    parser.add_argument('--data_type', type=str, default='smart')
    parser.add_argument('--data_root', type=str, default='data')
    parser.add_argument('--output_dir', type=str, default='experiments')
    parser.add_argument('--model', type=str, default='allenai/unifiedqa-t5-base')
    parser.add_argument('--options', type=list, default=["A", "B", "C", "D", "E"])
    parser.add_argument('--epoch', type=int, default=20)
    parser.add_argument('--lr', type=float, default=5e-5)
    parser.add_argument('--bs', type=int, default=16)
    parser.add_argument('--input_len', type=int, default=512)
    parser.add_argument('--output_len', type=int, default=64)
    parser.add_argument('--logging_steps', type=int, default=50)
    parser.add_argument('--eval_steps', type=int, default=100)
    parser.add_argument('--eval_bs', type=int, default=16)
    parser.add_argument('--eval_acc', type=int, default=None, help='evaluate accumulation step')
    parser.add_argument('--train_split', type=str, default='train', choices=['train', 'trainval', 'minitrain'])
    parser.add_argument('--val_split', type=str, default='val', choices=['test', 'val', 'minival'])
    parser.add_argument('--test_split', type=str, default='test', choices=['test', 'minitest'])
    parser.add_argument('--use_generate', type=bool, default=True, help='only for baseline to improve inference speed')
    parser.add_argument('--final_eval', action='store_true', help='only evaluate the model at the final epoch')
    parser.add_argument('--user_msg', type=str, default="baseline", help='experiment type in the save_dir')
    parser.add_argument('--img_type', type=str, default="detr", choices=['detr', 'clip', 'resnet','vit'], help='type of image features')
    parser.add_argument('--eval_le', type=str, default=None, help='generated rationale for the dev set')
    parser.add_argument('--test_le', type=str, default=None, help='generated rationale for the test set')
    parser.add_argument('--evaluate_dir', type=str, default=None, help='the directory of model for evaluation')
    parser.add_argument('--caption_file', type=str, default='data/captions.json')
    parser.add_argument('--use_caption', action='store_true', help='use image captions or not')
    parser.add_argument('--seed', type=int, default=1123, help='random seed')
    
    parser.add_argument('--gpt_data_include_level', type=int, default=2)
    parser.add_argument('--GPT_augmentation_dict_path', type=str, default='./data/GT_rationale/smart/gpt_augmentation_result_total.json')
    
    args = parser.parse_args()
    return args
        
def T5Trainer(
    args,
):  

    os.environ["WANDB_PROJECT"] = args.proj_name
    wandb.init(project=args.proj_name, name=args.run_name)    
        
    torch.manual_seed(args.seed)  # pytorch random seed
    np.random.seed(args.seed)  # numpy random seed
    torch.backends.cudnn.deterministic = True
    
    if args.evaluate_dir is not None:
        args.model = args.evaluate_dir

    tokenizer = AutoTokenizer.from_pretrained(args.model)

    console.log(f"""[Model]: Loading {args.model}...\n""")
    console.log(f"[Data]: Reading data...\n")
    
    if args.evaluate_dir is not None:
        save_dir = args.evaluate_dir
    else:
        model_name = args.model.replace("/","-")
        gpu_count = torch.cuda.device_count()
        save_dir = f"{args.output_dir}/{args.user_msg}_{model_name}_{args.img_type}__lr{args.lr}_bs{args.bs * gpu_count}_op{args.output_len}_ep{args.epoch}"
        if not os.path.exists(save_dir):
            os.mkdir(save_dir)
    print(save_dir)

    patch_size = img_shape[args.img_type]
    model = T5ForMultimodalGeneration.from_pretrained(args.model, patch_size=patch_size) 
    print("\nmodel parameters: ", model.num_parameters())
    print(f'Model require grad parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}')
    
    name_maps, image_features, problems, qids = load_img_features(args)
    if args.data_type == 'smart':
        train_set = MMCOT_SMART_Dataset(args, name_maps, image_features, tokenizer, 'train')
        eval_set = MMCOT_SMART_Dataset(args, name_maps, image_features, tokenizer, 'valid')
        test_set = MMCOT_SMART_Dataset(args, name_maps, image_features, tokenizer, 'test')
        valid_gen = MMCOT_SMART_Dataset(args, None, None, tokenizer, 'gen_valid')
        test_gen = MMCOT_SMART_Dataset(args, None, None, tokenizer, 'gen_test')
    
    elif args.data_type == 'scienceqa':
        train_qid, valid_qid, test_qid = qids['train'], qids['val'][:20], qids['test'][:20]
        train_set = ScienceQADatasetImg(
            problems,
            train_qid,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            mode='train'
        )
        eval_set = ScienceQADatasetImg(
            problems,
            valid_qid,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.eval_le,
            mode='valid'
        )
        test_set = ScienceQADatasetImg(
            problems,
            test_qid,
            name_maps,
            tokenizer,
            args.input_len,
            args.output_len,
            args,
            image_features,
            args.test_le,
            mode='test'
        )
    datacollator = DataCollatorForSeq2Seq(tokenizer)    
    
    # rougel for rationale generation
    metric = evaluate.load("rouge")
    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def compute_metrics_rougel(eval_preds):
        if args.use_generate:
            preds, targets = eval_preds
            if isinstance(preds, tuple):
                preds = preds[0]
        else:
            preds = eval_preds.predictions[0]
            targets = eval_preds.label_ids
            preds = preds.argmax(axis=2)
        
        preds[preds == -100] = tokenizer.pad_token_id
        targets[targets == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(preds, skip_special_tokens=True, clean_up_tokenization_spaces=True)
        targets = tokenizer.batch_decode(targets, skip_special_tokens=True, clean_up_tokenization_spaces=True)

        decoded_preds, decoded_labels = postprocess_text(preds, targets)

        result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        result = {k: round(v * 100, 4) for k, v in result.items()}
        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        return result

    # only use the last model for evaluation to save time
    if args.final_eval:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            run_name = args.run_name,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=False,
            evaluation_strategy="no",
            logging_strategy="steps",
            save_strategy="epoch",
            logging_steps = args.logging_steps,
            eval_steps = args.eval_steps,
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            report_to="wandb",
        )
    # evaluate at each epoch
    else:
        training_args = Seq2SeqTrainingArguments(
            save_dir,
            run_name = args.run_name,
            do_train=True if args.evaluate_dir is None else False,
            do_eval=True,
            evaluation_strategy="epoch",
            logging_strategy="steps",
            save_strategy="epoch",
            logging_steps = args.logging_steps,
            eval_steps = args.eval_steps,
            save_total_limit = 2,
            learning_rate= args.lr,
            eval_accumulation_steps=args.eval_acc,
            per_device_train_batch_size=args.bs,
            per_device_eval_batch_size=args.eval_bs,
            weight_decay=0.01,
            num_train_epochs=args.epoch,
            metric_for_best_model="rougeL",
            predict_with_generate=args.use_generate,
            generation_max_length=args.output_len,
            load_best_model_at_end=True,
            report_to="wandb",
        )

    trainer = Seq2SeqTrainer(
        model=model,
        args=training_args,
        train_dataset=train_set,
        eval_dataset=eval_set,
        data_collator=datacollator,
        tokenizer=tokenizer,
        compute_metrics = compute_metrics_rougel
    )

    if args.evaluate_dir is None:
        print('\nMMCOT Rataionale Generation Started')
        trainer.train()
        trainer.save_model(save_dir)
    
    print('\nMMCOT Ratioanle Test Set Generation')    
    # metrics = trainer.evaluate(eval_dataset = test_set, max_length=args.output_len)
    # trainer.log_metrics("test", metrics)
    # trainer.save_metrics("test", metrics)

    predict_results = trainer.predict(test_dataset=test_set, max_length=args.output_len) 
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds[preds == -100] = tokenizer.pad_token_id
        targets[targets == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Save Test Set Rationale Generation Result
        result_json = dict()
        if args.data_type == 'scienceqa':
            for idx, qid in enumerate(test_qid):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                result_json[str(qid)] = {'pred':pred.strip(), 'label':ref.strip()}
        
        elif args.data_type == 'smart':
            answer_promtpt_len = len("Solution: ")
            for idx, (im_path, im_name, question) in enumerate(test_gen):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                result_json[im_name] = {'question': question, 'pred':pred[answer_promtpt_len:], 'label':ref}

        result_json_path = os.path.join(args.output_dir, "predictions_ans_test.json")
        with open(result_json_path, "w") as writer:
            json.dump(result_json, writer, ensure_ascii=False, indent=4)
        
    # generate the rationale for the eval set
    torch.cuda.empty_cache()
    del predict_results, preds, targets
    
    predict_results = trainer.predict(test_dataset=eval_set, max_length=args.output_len) 
    if trainer.is_world_process_zero():
        if args.use_generate:
            preds, targets = predict_results.predictions, predict_results.label_ids
        else:
            preds = predict_results.predictions[0]
            targets = predict_results.label_ids
            preds = preds.argmax(axis=2)

        preds[preds == -100] = tokenizer.pad_token_id
        targets[targets == -100] = tokenizer.pad_token_id
        preds = tokenizer.batch_decode(
            preds, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        targets = tokenizer.batch_decode(
            targets, skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        
        # Save Test Set Rationale Generation Result
        result_json = dict()
        if args.data_type == 'scienceqa':
            for idx, qid in enumerate(valid_qid):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                result_json[str(qid)] = {'pred':pred.strip(), 'label':ref.strip()}
        
        elif args.data_type == 'smart':
            answer_promtpt_len = len("Solution: ")
            for idx, (im_path, im_name, question) in enumerate(valid_gen):
                pred = preds[int(idx)]
                ref = targets[int(idx)]
                result_json[im_name] = {'question': question, 'pred':pred[answer_promtpt_len:], 'label':ref}

        result_json_path = os.path.join(args.output_dir, "predictions_ans_eval.json")
        with open(result_json_path, "w") as writer:
            json.dump(result_json, writer, ensure_ascii=False, indent=4)

if __name__ == '__main__':

    # training logger to log training progress
    training_logger = Table(
        Column("Epoch", justify="center"),
        Column("Steps", justify="center"),
        Column("Loss", justify="center"),
        title="Training Status",
        pad_edge=False,
        box=box.ASCII,
    )
    
    args = parse_args()
    print("args",args)
    print('====Input Arguments====')
    print(json.dumps(vars(args), indent=2, sort_keys=False))
    gv.custom_globals_init()
    random.seed(args.seed)
    
    if not os.path.exists(args.output_dir):
            os.mkdir(args.output_dir)

    T5Trainer(
        args = args
    )
