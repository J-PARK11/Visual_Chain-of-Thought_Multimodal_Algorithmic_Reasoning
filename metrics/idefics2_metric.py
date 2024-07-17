from dataclasses import dataclass

@dataclass
class base_metric:
    def __init__(self, model_args, data_args, processor, info):
        self.processor = processor
    
    def compute_metrics(self, pred):
        print(pred)
        print(pred.keys())
        metric = {'Acc': 100}
        return metric
        
