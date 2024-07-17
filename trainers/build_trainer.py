from .idefics2_trainer import Trainer
# from transformers import Trainer

def get_trainer(model_args, training_args, model, processor, data_module, metric=None):
    if model_args.model_type == "Idefics2-8b":
        trainer = Trainer(
            model=model,
            args=training_args,
            data_collator=data_module['data_collator'],
            train_dataset=data_module['train_dataset'],
            eval_dataset=data_module['val_dataset'],
            # compute_metrics=metric.compute_metrics
            )
    else:
        raise NotImplementedError
    return trainer