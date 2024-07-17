# DDP run script
export CUDA_VISIBLE_DEVICES=1,2,3,4,5,6,7
torchrun --nproc_per_node 7 train.py \
    --run_name full_sentence_ft \
    --output_dir ./checkpoints/full_sentence_ft/ \
    --num_train_epochs 2 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --gradient_accumulation_steps 1 \
    --report_to wandb \
    --logging_steps 10 \
    --evaluation_strategy steps \
    --eval_steps 20 \
    --save_strategy steps \
    --save_steps 1000 \
    --save_total_limit 20