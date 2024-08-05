import torch
from peft import LoraConfig
# from transformers import AutoProcessor, BitsAndBytesConfig, Idefics2ForConditionalGeneration#, AwqConfig

from transformers import BitsAndBytesConfig
from .Idefics2.processing_idefics2 import Idefics2Processor
from .Idefics2.modeling_idefics2 import Idefics2ForConditionalGeneration

def get_model(mode, data_args, model_args, training_args):

    if model_args.model_type == "Idefics2-8b":
        processor = Idefics2Processor.from_pretrained(model_args.pretrained_model_path,
                                                  do_image_splitting = model_args.do_image_splitting,
                                                  size= {"longest_edge": 448, "shortest_edge": 378})
                                                  # size= {"longest_edge": 224, "shortest_edge": 190}   
        
        if 'train' in mode:
            if model_args.USE_LORA :
                lora_config = LoraConfig(
                    r=model_args.lora_r,
                    lora_alpha=model_args.lora_alpha,
                    lora_dropout=model_args.lora_dropout,
                    target_modules='.*(text_model|modality_projection|perceiver_resampler).*(down_proj|gate_proj|up_proj|k_proj|q_proj|v_proj|o_proj).*$',
                    use_dora=False if model_args.USE_QLORA else True,
                    init_lora_weights="gaussian"
                )
                
                if model_args.USE_QLORA:
                    BnB_config = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_compute_dtype=torch.bfloat16
                    )
            
            if model_args.load_ckpt_path:                                                
                model = Idefics2ForConditionalGeneration.from_pretrained(model_args.load_ckpt_path, 
                                                                    torch_dtype=torch.bfloat16,
                                                                    quantization_config=BnB_config if model_args.USE_QLORA else None,
                                                                    max_length = model_args.max_length, # 20
                                                                    low_cpu_mem_usage=True)
                print(f'Load ckpt: {model_args.load_ckpt_path}')
                
            else:
                model = Idefics2ForConditionalGeneration.from_pretrained(model_args.pretrained_model_path,
                                                                    torch_dtype=torch.bfloat16,
                                                                    quantization_config=BnB_config if model_args.USE_QLORA else None,
                                                                    max_length = model_args.max_length, # 20
                                                                    low_cpu_mem_usage=True
                                                                    # quantization_config=default_quantization_config
                                                                    # _attn_implementation="flash_attention_2"
                                                                    )#.to(training_args.device)
            if model_args.USE_LORA :
                if model_args.load_ckpt_path:
                    adapter_name = 'second_train'
                else:
                    adapter_name = 'first_train'
                    model.add_adapter(lora_config, adapter_name=adapter_name)
                    model.enable_adapters()
        
        else: # valid, test
            if model_args.load_ckpt_path:                                                
                model = Idefics2ForConditionalGeneration.from_pretrained(model_args.load_ckpt_path, 
                                                                        torch_dtype=torch.bfloat16,
                                                                        max_length=model_args.max_length,
                                                                        low_cpu_mem_usage=True)            
            else:
                model = Idefics2ForConditionalGeneration.from_pretrained(model_args.pretrained_model_path, 
                                                                    torch_dtype=torch.bfloat16,
                                                                    max_length=model_args.max_length,
                                                                    low_cpu_mem_usage=True)     
            
    else:
        raise NotImplementedError

    print(f'\nModel Type: {model_args.model_type}, mode: {mode}, load_ckpt_path: {model_args.load_ckpt_path}')
    return model, processor

# default_quantization_config = AwqConfig(
#     bits=4,
#     fuse_max_seq_len=4096,
#     modules_to_fuse={
#         "attention": ["q_proj", "k_proj", "v_proj", "o_proj"],
#         "mlp": ["gate_proj", "up_proj", "down_proj"],
#         "layernorm": ["input_layernorm", "post_attention_layernorm", "norm"],
#         "use_alibi": False,
#         "num_attention_heads": 32,
#         "num_key_value_heads": 8,
#         "hidden_size": 4096,
#      }
# )