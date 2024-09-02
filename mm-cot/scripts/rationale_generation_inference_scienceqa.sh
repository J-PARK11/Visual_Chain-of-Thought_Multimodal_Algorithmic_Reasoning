# rationale generation: SMART
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --proj_name scienceqa_rationale_generation_inference_project \
    --run_name scienceqa_rationale_generation_inference_run \
    --data_root /data/ScienceQA  --data_type scienceqa \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type vit \
    --bs 2 --eval_bs 4 --output_len 512 \
    --use_generate True \
    --output_dir ./data/generated_rationale/scienceqa/ \
    --evaluate_dir ./checkpoints/pretrained/large-rationale