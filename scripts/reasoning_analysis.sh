# Only Single GPU
python V_COT_reasoning_analysis.py \
    # --num_workers 8 \
    # --batch_size 1 \
    # --data_tot 3 \
    # --train_puzzle_list "1,2,6,7,17,19,40,77" \
    # --gpu_num 0 \
    # --VLM_type Idefics2 \
    # --data_root /data/SMART101-release-v1/SMART101-Data/ \ 
    # --save_root ./V_COT_output/