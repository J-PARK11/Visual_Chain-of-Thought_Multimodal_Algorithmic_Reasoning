# DDP run script
export CUDA_VISIBLE_DEVICES=0
torchrun --nproc_per_node 1 train.py \
    --run_name gt_101_rationale_DPR_train \
    --output_dir ./checkpoints/gt_101_rationale_DPR_train/ \
    --task GT_with_rationale \
    --train_tot 1 \
    --eval_tot 3 \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_strategy epoch \
    --save_steps 1500 \
    --save_total_limit 20 \
    --val_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 \
    --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 \
    --USE_DPR True