# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 train.py \
    --run_name GPT_aug_level2_train \
    --output_dir ./checkpoints/GPT_aug_level2_train/ \
    --task GPT_augmentation_train \
    --train_tot 1000 \
    --eval_tot 3 \
    --num_train_epochs 3 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 2 \
    --report_to wandb \
    --logging_steps 20 \
    --evaluation_strategy steps \
    --eval_steps 40 \
    --save_strategy epoch \
    --save_steps 1500 \
    --save_total_limit 20 \
    --gpt_data_include_level 2 \
    --val_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 \
    --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77