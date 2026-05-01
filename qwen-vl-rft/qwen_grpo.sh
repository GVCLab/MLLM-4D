export MODELSCOPE_OFFLINE=1
export HF_DATASETS_OFFLINE=1
export TRANSFORMERS_OFFLINE=1
export FPS_MAX_FRAMES=32

MODEL=PATH-to-MLLM-4D-SFT-CoT
REWORD=PATH-to-ST-reward
DATA=PATH-to-MLLM4D-R1-30k.jsonl
OUTPUT=output/grop_output

CUDA_VISIBLE_DEVICES=1,2,3,4,5,6 \
NPROC_PER_NODE=6 \
MAX_PIXELS=602112 \
swift rlhf \
    --rlhf_type grpo \
    --model ${MODEL} \
    --external_plugins ${REWORD} \
    --reward_funcs format_reward_ours acc_reward_ours camera_reward_ours object_reward_ours \
    --use_vllm true \
    --vllm_mode server \
    --vllm_server_host 127.0.0.1 \
    --vllm_server_port 8000 \
    --train_type lora \
    --torch_dtype bfloat16 \
    --dataset ${DATA} \
    --max_completion_length 4096 \
    --num_train_epochs 1 \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 4 \
    --dataloader_num_workers 0 \
    --learning_rate 1e-4 \
    --gradient_accumulation_steps 1 \
    --save_strategy 'steps' \
    --eval_strategy 'steps' \
    --eval_steps 2000 \
    --save_steps 2000 \
    --save_total_limit 5 \
    --logging_steps 10 \
    --output_dir ${OUTPUT} \
    --warmup_ratio 0.001 \
    --num_generations 12 \
    --generation_batch_size 24 \
    --temperature 1.0 \
    --log_completions true \
    --async_generate true \
    --report_to tensorboard \
    --beta 0.001 \
