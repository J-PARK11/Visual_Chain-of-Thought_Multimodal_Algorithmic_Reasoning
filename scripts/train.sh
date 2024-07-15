# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7
torchrun --nproc_per_node 8 train.py \
    --run_name Zero_Shot_train \
    --output_dir ./checkpoints/Zero_Shot_train/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 8 \
    --per_device_eval_batch_size 8 \
    --gradient_accumulation_steps 2 \
    --report_to wandb \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 100 \
    --save_strategy steps \
    --save_steps 500 \
    --save_total_limit 20