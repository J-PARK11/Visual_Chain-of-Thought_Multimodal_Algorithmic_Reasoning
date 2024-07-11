# DDP run script
export CUDA_VISIBLE_DEVICES=0,1
torchrun --nproc_per_node 2 train.py \
    --output_dir ./checkpoints/dump/ \
    --num_train_epochs 5 \
    --per_device_train_batch_size 1