# DDP run script
export CUDA_VISIBLE_DEVICES=1
torchrun --nproc_per_node 1 eval.py \
    --run_name evaluation \
    --output_dir ./checkpoints/evaluation/ \
    --task supervised \
    --train_tot 1000 \
    --eval_tot 3 \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 10 \
    --save_strategy epoch \
    --save_steps 1500 \
    --save_total_limit 20 