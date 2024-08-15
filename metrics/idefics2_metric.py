import copy

from dataclasses import dataclass
import transformers
from transformers import AutoProcessor, PreTrainedModel

import numpy as np
import torch
from torch.nn import CosineSimilarity

from utils.util import is_float, read_dataset_info


@dataclass
class base_metric:
    def __init__(self, model_args, data_args, processor, info):
        self.processor = processor
    
    def compute_metrics(self, pred):
        print(pred)
        print(pred.keys())
        metric = {'Acc': 100}
        return metric

@dataclass   
class ComputeMetricAnswerKey:

    processor: transformers.AutoProcessor
    b_pids: list
    vicuna_embedding : transformers.PreTrainedModel
    
    def __post_init__(self):

        self.lower_candidates = {
            "a" : "A",
            "b" : "B",
            "c" : "C",
            "d" : "D",
            "e" : "E"
        }
        self.cossim = CosineSimilarity(dim=1)

        # 어떤 test 요소가 
        # Test 시의 실제 test 요소 순서대로  list가 필요함 -> self.qa_info를 copy해놓고 pid들만 추출

        # 순서대로 pid들만 추출해야함
        self.puzzle_path = "/home/work/g-earth-22/VLM/database/ORIGINAL/SMART-101/data/SMART101-release-v1/puzzle_type_info.csv"
        self.puzzles = read_dataset_info(self.puzzle_path)


    def compute_metrics(self, pred):
        pred.answers[pred.answers == -100] = self.processor.tokenizer.pad_token_id
        tmp_gt_answer_list = self.processor.tokenizer.batch_decode(pred.answers, skip_special_tokens=True)
        # gT answer list 가 input처럼 나옴 
        # 'User: Looking at this image, solve the question. Question: The plates pictured are painted to two cartons. The numbers in each carton add to the same number. Which number must be in the carton with the number 6?\nOptions:\nA. 13\nB. 19\nC. 1\nD. 11\nE. 6\nPlease answer with the alphabet in the options. \nAssistant: Answer: B'
        gt_answer_list = []
        for each_tmp_gt in tmp_gt_answer_list:
            gt_answer_list.append(each_tmp_gt.split("Answer: ")[-1])

        pred.predictions[pred.predictions == -100] = self.processor.tokenizer.pad_token_id
        pred_answer_list = self.processor.tokenizer.batch_decode(pred.predictions, skip_special_tokens=True)

        option_value = ["A","B","C","D","E"]

        # 근사하는 과정...
        # 1. 풀이과정과 answer을 분리하는 과정 필요 -> 만약 "Answer is" 하는 부분이 없다면?
        parser = "Answer: "
        parsed_answer = []
        for each_pred_answer in pred_answer_list:
            each_parsed_answer = each_pred_answer.split(parser)[-1]
            # parse한게 소문자일수도 있고 대문자일수도 있고 blank가 들어있을 수 있고 등등.. => 이후에 cosine similarity 기준으로 비교
            parsed_answer.append(each_parsed_answer)
        # breakpoint()
        assert len(parsed_answer) == len(gt_answer_list)

        self.processor.tokenizer.add_bos_token=False
        non_approximated_pred=[]
        approximated_pred=[]
        print(f"pred answer list : {len(pred_answer_list)}")

        for idx, each_pred in enumerate(parsed_answer):
            each_pred = str(each_pred).strip()
            gt_answer = gt_answer_list[idx].strip()

              #소문자인경우 대문자로 변경
            if each_pred in self.lower_candidates.keys():
                each_pred = self.lower_candidates[each_pred]
            #for s_acc
            if each_pred == gt_answer:
                non_approximated_pred.append(True)
            else:
                non_approximated_pred.append(False)
            
            #approximation
            #'A','B','C','D','E' 와 비교
            option_value = ['A','B','C','D','E']
            # breakpoint()
            option_tokenized = self.processor.tokenizer(text = option_value, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[5, seqlen, 4096]
            #NOTE : padding시 padding token의 embedding은 어떻게 되는지 보기
            option_embedded = self.vicuna_embedding(option_tokenized).mean(axis=1) #[5, 4096]

            each_pred_tokenized = self.processor.tokenizer(text=each_pred, padding=True, truncation=False, return_tensors="pt").input_ids.long() #[1, seqlen, 4096]
            each_pred_embedded = self.vicuna_embedding(each_pred_tokenized).mean(axis=1) #[1, 4096]

            approximated_option_index = self.cossim(option_embedded, each_pred_embedded).argmax(dim=0)
            result = (option_value[approximated_option_index] == gt_answer)
            approximated_pred.append(result)

        # breakpoint()
        assert len(approximated_pred) == len(pred_answer_list) == pred.predictions.shape[0]
        print(f"non approximated pred : {len(non_approximated_pred)}")
        #calculate s_acc/o_acc & puzzle_id 
        tot_samples_num = pred.predictions.shape[0]
        puzzle_acc = {}
        for t in list(set(self.b_pids)):
            puzzle_acc[str(t)] = [
                np.array(non_approximated_pred)[np.array(self.b_pids) == t].sum(),
                np.array(approximated_pred)[np.array(self.b_pids) == t].sum(),
                (np.array(self.b_pids) == t).sum()
            ]

        to_int = lambda x: np.array(list(x)).astype("int")
        cls_mean = lambda x, idx, pids: np.array([x[int(ii)] for ii in idx]).sum() / len(
            set(to_int(idx)).intersection(set(to_int(pids)))
        )
        acc_list = np.zeros(101+1)
        opt_acc_list = np.zeros(101+1)
        for puzzle_id in puzzle_acc.keys():
            acc = 100.0 * puzzle_acc[puzzle_id][0] / puzzle_acc[puzzle_id][2]
            oacc = 100.0 * puzzle_acc[puzzle_id][1] / puzzle_acc[puzzle_id][2]
            acc_list[int(puzzle_id)] = acc
            opt_acc_list[int(puzzle_id)] = oacc
        #print acc, opt_acc by puzzle id
        for t in range(1, 101+1):
            print("%d opt_acc(%%)=%0.2f acc(%%)=%0.2f" % (t, opt_acc_list[t], acc_list[t]), end="\t")
            if t % 5 == 0:
                print("\n")
        print("\n\n")
        class_avg_perf = {}
        classes = ["counting", "math", "logic", "path", "algebra", "measure", "spatial", "pattern"]
        print(classes)
        for each_class in classes:
            idx_list = self.puzzles[each_class]
            class_avg_perf[each_class] = (
                cls_mean(acc_list, idx_list, list(puzzle_acc.keys())),
                cls_mean(opt_acc_list, idx_list, list(puzzle_acc.keys())),
            )
            print("%0.1f/%0.1f & " % (class_avg_perf[each_class][0], class_avg_perf[each_class][1]), end=" ")
        print("\n\n")

        metrics = {
            "S_acc" : np.array(non_approximated_pred).sum()*100 / tot_samples_num,
            "O_acc" : np.array(approximated_pred).sum()*100 / tot_samples_num,
        }
        #result에 class별 s_acc / o_acc append 혹은 update 
        for each_class in classes:
            metrics[f"{each_class}_acc"] = class_avg_perf[each_class][0]
            metrics[f"{each_class}_oacc"] = class_avg_perf[each_class][1]

        # #category acc
        # metrics[f"category_acc"] = category_accuracy

        #원상복구
        self.processor.tokenizer.add_bos_token=True

        return metrics
