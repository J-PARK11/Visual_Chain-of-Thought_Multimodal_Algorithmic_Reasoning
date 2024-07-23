# Only Single GPU
python inference.py \
    # --num_workers 8 \
    # --batch_size 1 \
    # --data_tot 3 \
    # --train_puzzle_list "1,2,6,7,17,19,40,77" \
    # --gpu_num 0 \
    # --VLM_type Idefics2 \
    # --data_root /data/SMART101-release-v1/SMART101-Data/ \ 
    # --save_root ./V_COT_output/

python inference.py --output_name GT_with_rationale-20.json --load_ckpt_path checkpoints/GT_with_rationale/checkpoint-20 --test_puzzle_list 1,2,6,7,10,17,19,40,77 --gpu_num 1 --experiment_number 1,2,3,6