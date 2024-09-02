# rationale generation: SMART
CUDA_VISIBLE_DEVICES=0,1,2,3 python main.py \
    --proj_name mmcot_rationale_generation_inference_project \
    --run_name mmcot_rationale_generation_inference_run \
    --data_root /data/SMART101-release-v1/SMART101-Data \
    --model declare-lab/flan-alpaca-large \
    --user_msg rationale --img_type detr \
    --bs 2 --eval_bs 4 --output_len 512 \
    --use_generate True \
    --output_dir ./data/generated_rationale/smart/ \
    --evaluate_dir ./checkpoints/rationale_generation_mmcot_smart/rationale_declare-lab-flan-alpaca-large_detr__lr5e-05_bs8_op512_ep1/checkpoint-12