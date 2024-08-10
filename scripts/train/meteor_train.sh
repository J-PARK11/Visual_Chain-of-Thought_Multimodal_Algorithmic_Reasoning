# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc_per_node 3 train.py \
    --run_name METEOR_3000h_FT \
    --output_dir ./checkpoints/METEOR_3000h_FT/ \
    --task METEOR_FT \
    --data_type METEOR \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --lora_dropout 0.2 \
    --report_to wandb \
    --logging_steps 50 \
    --save_strategy steps \
    --save_steps 10000 \
    --save_total_limit 30