# Only Single GPU

# Baseline
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Baseline_acc_test.json --eval_tot 300 --gpu_num 0

# Option FT
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Option_FT_acc_test.json --load_ckpt_path checkpoints/loss_optimization/checkpoint-102 --eval_tot 300 --gpu_num 1

# GPT aug 2000
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_2000_acc_test.json --load_ckpt_path checkpoints/GPT_augmentation/checkpoint-672 --eval_tot 300 --gpu_num 2

# GPT aug Level2
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_acc_test.json --load_ckpt_path checkpoints/GPT_aug_level2_train/checkpoint-11125 --eval_tot 300 --gpu_num 2

# Phase2_DPR_CA
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Phase2_DPR_CA_acc_test.json --load_ckpt_path checkpoints/phase2_DPR_CA/checkpoint-0000 --eval_tot 300 --gpu_num 3

# dump
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_1000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-1000 --eval_tot 3 --gpu_num 0
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_3000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-3000 --eval_tot 3 --gpu_num 1
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_5000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-5000 --eval_tot 3 --gpu_num 2
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_7000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-7000 --eval_tot 3 --gpu_num 3

python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_16000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-16000 --eval_tot 3 --gpu_num 0
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_18000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-18000 --eval_tot 3 --gpu_num 1
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_20000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-20000 --eval_tot 3 --gpu_num 2
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_cpkt_22000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot/checkpoint-22000 --eval_tot 3 --gpu_num 3

python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name gt_101_rationale_DPR_cpkt_101.json --load_ckpt_path checkpoints/gt_101_rationale_DPR/checkpoint-101 --eval_tot 3 --gpu_num 0

python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_dev_cpkt_2000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot_dev/checkpoint-2000 --eval_tot 3 --gpu_num 0
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_level2_DPR_qkv_reverse_1000tot_dev_cpkt_4000.json --load_ckpt_path checkpoints/GPT_aug_level2_DPR_qkv_reverse_1000tot_dev/checkpoint-4000 --eval_tot 3 --gpu_num 1