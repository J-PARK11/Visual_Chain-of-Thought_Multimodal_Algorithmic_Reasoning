import random
from .V_COT_data_loader import V_COT_SMART101_Dataset

def get_dataset(training_args, model_args, data_args, processor=None):
    train_dataset = V_COT_SMART101_Dataset(data_args, 'train')
    val_dataset = V_COT_SMART101_Dataset(data_args, 'valid')
    test_dataset = V_COT_SMART101_Dataset(data_args, 'test')
    collator = V_COT_collator(processor)
    
    print(f'\nTrain Dataset: {len(train_dataset)}')
    print(f'Valid Dataset: {len(val_dataset)}, {data_args.val_puzzle_list}')
    print(f'Test Dataset: {len(test_dataset)}, {data_args.test_puzzle_list}')
        
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
            text = self.processor.apply_chat_template(messages, add_generation_prompt=False)
            texts.append(text.strip())
            images.append([image])

        batch = self.processor(text=texts, images=images, return_tensors="pt", padding=True)

        labels = batch["input_ids"].clone()
        labels[labels == self.processor.tokenizer.pad_token_id] = -100
        labels[labels == self.image_token_id] = -100
        # labels[:,:-5] = -100
        batch["labels"] = labels

        return batch
