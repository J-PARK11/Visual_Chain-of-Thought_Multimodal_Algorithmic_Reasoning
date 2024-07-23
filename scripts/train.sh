# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 train.py \
    --run_name GT_with_rationale_15_shot \
    --output_dir ./checkpoints/GT_with_rationale_15_shot/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 200 \
    --save_strategy steps \
    --save_steps 1500 \
    --save_total_limit 20