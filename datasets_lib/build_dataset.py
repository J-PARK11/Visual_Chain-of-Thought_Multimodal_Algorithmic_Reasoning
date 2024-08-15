import random
from .V_COT_data_loader import V_COT_SMART101_Dataset

def get_dataset(training_args, model_args, data_args, processor=None):
    train_dataset = V_COT_SMART101_Dataset(data_args, 'train')
    val_dataset = V_COT_SMART101_Dataset(data_args, 'valid')
    test_dataset = V_COT_SMART101_Dataset(data_args, 'test')
    collator = V_COT_collator(processor)
    eval_collator = V_COT_eval_collator(processor)
    
    print(f'\nTrain Dataset: {len(train_dataset)}')
    print(f'Valid Dataset: {len(val_dataset)}, {data_args.val_puzzle_list}')
    print(f'Test Dataset: {len(test_dataset)}, {data_args.test_puzzle_list}')
        
    return dict(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, data_collator=collator, eval_collator=eval_collator)
    
class V_COT_collator: #NOTE : use for traininig
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
    
    def __call__(self, examples):

        texts = []
        images = []
        for idx, (image, im_p, pids, question, opts, answer_option, lbl, answer) in enumerate(examples):
            question = "Question: " + question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'Looking at this image, solve the question.'},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_option}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False) #-> True 면 맨 뒤에 '\nAssistant:' 이렇게 추가로 붙음
            # Assistant 전후로 <end_of_utterance> token 추가
            """
            'User: Looking at this image, solve the question.<image>Question: The sticks lie on top of each other. Stick 3 is at the bottom. Stick 6 is at the top. 
            The stick in the middle is:\nOptions:\nA. 1\nB. 7\nC. 5\nD. 2\nE. 4\nPlease answer with the alphabet in the options and explain how you solved it.
            <end_of_utterance>\nAssistant: We need to identify the stick in the middle.\nFirst, stick 3 is currently at the bottom and stick 6 is at the top. 
            Observing the rest of the sticks based on this, they are stacked in the order of 3-1-4-7-2-5-6. So the middle one is stick 7.
            \nTherefore, the answer is B=7.<end_of_utterance>\n'
            """
            texts.append(text.strip())
            images.append([image])

        # eval 때는 풀이과정이 답이니까 뒤에 붙혀야함 
        # batch key : "input_ids", "attention_mask", "pixel_values", "pixel_attention_mask"
        # 여기서 내부적으로 attention_mask를 어떻게 넣고 있는지 확인해야함
        # pixel_attention_mask size?? [1, 5, 448, 448] -> 5가 뭐지ㅇ..? 언제 apply 되는거지 patch mask가 아니라 pixel_attentionmask?
        # => padding pixel_attention_mask에 attend 하는 것을 막음. => [B, C, H, W]인가? 문서에는 [B, #image, C, H, W] 인데 C가 5인것도 이상함
        # input_ids가 [1, 510] -> tokenize를 어떻게 했길래..?
        # attention_mask -> 여기서는 padding mask 만 반영된 것
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        batch["labels"] = labels

        return batch




# Inference 때는 msg에 assistant 가 안들어가게 넣고, batch labels을 따로 만들어 줘야함

  
class V_COT_eval_collator: #NOTE : use for traininig
    """
        https://github.com/NielsRogge/Transformers-Tutorials/blob/master/Idefics2/Fine_tune_Idefics2_for_multi_page_PDF_question_answering_on_DUDE.ipynb
    """
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
    
    def __call__(self, examples):

        texts = []
        images = []
        answers = []
        for idx, (image, im_p, pids, question, opts, answer_option, lbl, answer) in enumerate(examples):
            question = "Question: " + question
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'Looking at this image, solve the question.'},
                        {"type": "image"},
                        {"type": "text", "text": question}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=True)
            # Assistant 전후로 <end_of_utterance> token 추가
            """
            'User: Looking at this image, solve the question.<image>Question: The sticks lie on top of each other. Stick 3 is at the bottom. Stick 6 is at the top. 
            The stick in the middle is:\nOptions:\nA. 1\nB. 7\nC. 5\nD. 2\nE. 4\nPlease answer with the alphabet in the options and explain how you solved it.
            <end_of_utterance>\nAssistant: We need to identify the stick in the middle.\nFirst, stick 3 is currently at the bottom and stick 6 is at the top. 
            Observing the rest of the sticks based on this, they are stacked in the order of 3-1-4-7-2-5-6. So the middle one is stick 7.
            \nTherefore, the answer is B=7.<end_of_utterance>\n'
            """
            texts.append(text.strip())
            images.append([image])
            answers.append(answer_option)

        # breakpoint()
        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)
        #eval 때는 label 앞을 다 padding 처리해야되나?

        # labels = self.processor(text=answers, return_tensors="pt", padding=True)
        # labels = batch["input_ids"].clone()
        # labels[labels == self.processor.tokenizer.pad_token_id] = -100
        # labels[labels == self.image_token_id] = -100
        # batch["labels"] = labels
        answer_batch = self.processor(text=answers, return_tensors="pt", padding=True)
        batch["answers"] = answer_batch["input_ids"]

        # labels가 없으면 compute_metrics 안에 안들어가짐 -> 1. 왜 idefics2는 이런식으로 동작하는가...-> 우선은 trainer 쪽에서 수정
        return batch

