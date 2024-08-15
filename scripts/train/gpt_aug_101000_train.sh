# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5
torchrun --nproc_per_node 6 train.py \
    --proj_name VCOT_HYKT \
    --run_name GPT_aug_level2_300eval_tot \
    --output_dir ./checkpoints/GPT_aug_level1_100tot/ \
    --task GPT_augmentation_train \
    --train_tot 1000 \
    --eval_tot 300 \
    --USE_QLORA false \
    --num_train_epochs 2 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --gradient_accumulation_steps 2 \
    --report_to wandb \
    --logging_steps 40 \
    --evaluation_strategy steps \
    --eval_steps 300 \
    --save_strategy no \
    --dataloader_num_workers 18 \
    --save_steps 1500 \
    --save_total_limit 20 \
    --gpt_data_include_level 1 \
    --val_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 \
    --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77


# export CUDA_VISIBLE_DEVICES=0
# python train.py \
#     --proj_name VCOT_HYKT \
#     --run_name GPT_aug_level2_300eval_tot \
#     --output_dir ./checkpoints/GPT_aug_level1_100tot/ \
#     --gpt_data_include_level 1 \
#     --task GPT_augmentation_train \
#     --train_tot 1000 \
#     --eval_tot 3 \
#     --USE_QLORA false \
#     --num_train_epochs 2 \
#     --per_device_train_batch_size 3 \
#     --per_device_eval_batch_size 3 \
#     --gradient_accumulation_steps 2 \
#     --report_to wandb \
#     --logging_steps 10 \
#     --evaluation_strategy steps \
#     --eval_steps 2 \
#     --save_strategy no \
#     --save_steps 1500 \
#     --save_total_limit 20 \
#     --gpt_data_include_level 2 \
#     --val_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 \
#     --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77