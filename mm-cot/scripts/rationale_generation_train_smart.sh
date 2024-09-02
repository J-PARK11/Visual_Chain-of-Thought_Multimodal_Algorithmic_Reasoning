# rationale generation: SMART
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --data_root /data/SMART101-release-v1/SMART101-Data \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type detr \
    --bs 2 --eval_bs 4  --epoch 2 --lr 5e-7 --output_len 512 \
    --use_generate True \
    --logging_steps 25 --eval_steps 100 \
    --output_dir ./checkpoints/rationale_generation_mmcot_smart