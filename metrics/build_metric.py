from .idefics2_metric import base_metric, ComputeMetricAnswerKey
import copy 

def get_metric(model_args, data_args, processor, data_module, model,info='base'):
    if info == 'base':
        # return base_metric(model_args, data_args, processor, info)
        # return ComputeMetricAnswerKey(model_args, data_args, processor, info)
        return ComputeMetricAnswerKey(processor, data_module["val_dataset"].pids, copy.deepcopy(model.model.text_model.get_input_embeddings()))