# DDP run script
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node 1 train.py \
    --run_name option_train_supervised \
    --output_dir ./checkpoints/option_train_supervised/ \
    --task supervised \
    --train_tot 1000 \
    --eval_tot 3 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_strategy epoch \
    --save_steps 1500 \
    --save_total_limit 20