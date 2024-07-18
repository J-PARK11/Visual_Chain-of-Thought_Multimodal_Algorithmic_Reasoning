# DDP run script
export CUDA_VISIBLE_DEVICES=0,1,2,3
torchrun --nproc_per_node 4 eval.py \
    --output_dir ./checkpoints/dump/ \
    --num_train_epochs 1 \
    --per_device_train_batch_size 1 \
    --per_device_eval_batch_size 1 \
    --report_to None