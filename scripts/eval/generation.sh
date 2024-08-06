# Only Single GPU

# Baseline
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Baseline_acc_test.json --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 --eval_tot 300 --gpu_num 0

# Option FT
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Option_FT_acc_test.json --load_ckpt_path checkpoints/loss_optimization/checkpoint-102 --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 --eval_tot 300 --gpu_num 1

# GPT aug 2000
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_2000_acc_test.json --load_ckpt_path checkpoints/GPT_augmentation/checkpoint-672 --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 --eval_tot 300 --gpu_num 2

# GPT aug Level2
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name GPT_aug_Level2_acc_test.json --load_ckpt_path checkpoints/GPT_aug_level2_train/checkpoint-22250 --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 --eval_tot 300 --gpu_num 3

# Phase2_DPR_CA
python generation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ablation_study/ --output_name Phase2_DPR_CA_acc_test.json --load_ckpt_path checkpoints/phase2_DPR_CA/checkpoint-0000 --test_puzzle_list 94,95,96,97,98,99,101,61,62,65,66,67,69,70,71,72,73,74,75,76,77 --eval_tot 300 --gpu_num 4