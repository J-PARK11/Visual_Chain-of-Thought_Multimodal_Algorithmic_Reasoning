from .idefics2_metric import base_metric

def get_metric(model_args, data_args, processor, info='base'):
    if info == 'base':
        return base_metric(model_args, data_args, processor, info)