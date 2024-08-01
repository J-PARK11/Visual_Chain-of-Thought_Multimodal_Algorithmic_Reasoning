# Only Single GPU
python GPT_augmentation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ --output_name gpt_augmentation_result --train_tot 1000 --phase 1_4
python GPT_augmentation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ --output_name gpt_augmentation_result --train_tot 1000 --phase 2_4
python GPT_augmentation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ --output_name gpt_augmentation_result --train_tot 1000 --phase 3_4
python GPT_augmentation.py --batch_size 1 --num_workers 8 --save_root ./V_COT_output/ --output_name gpt_augmentation_result --train_tot 1000 --phase 4_4