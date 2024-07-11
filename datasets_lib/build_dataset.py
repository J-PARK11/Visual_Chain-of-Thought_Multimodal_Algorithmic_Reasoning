import random
from .V_COT_data_loader import V_COT_reasoning_anlysis_dataset

def get_dataset(training_args, model_args, data_args, processor=None):
    if training_args.mode == 'few_shot_train':
        train_dataset = V_COT_reasoning_anlysis_dataset(data_args, 'train')
        val_dataset = V_COT_reasoning_anlysis_dataset(data_args, 'valid')
        test_dataset = V_COT_reasoning_anlysis_dataset(data_args, 'test')
        collator = V_COT_collator(processor)
        
        return dict(train_dataset=train_dataset, val_dataset=val_dataset, test_dataset=test_dataset, data_collator=collator)
    
class V_COT_collator:
    def __init__(self, processor):
        self.processor = processor
        self.image_token_id = processor.tokenizer.additional_special_tokens_ids[
            processor.tokenizer.additional_special_tokens.index("<image>")]
    
    def __call__(self, examples):
        texts = []
        images = []
        for idx, (image, im_p, pids, question, opts, answer_option, lbl, answer, answer_sheet) in enumerate(examples):
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": 'Looking at this image, solve the question.'},
                        {"type": "image"},
                        {"type": "text", "text": 'Question: {question}'}
                    ]
                },
                {
                    "role": "assistant",
                    "content": [
                        {"type": "text", "text": answer_option}
                    ]
                }
            ]
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = self.image_token_id
        batch["labels"] = labels

        return batch
