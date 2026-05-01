MERGED_MODEL_DIR=PATH-to-MLLM-4D-SFT-CoT

CUDA_VISIBLE_DEVICES=0 MAX_PIXELS=602112 swift rollout \
   --model ${MERGED_MODEL_DIR} \
   --max_new_tokens 4096 \
   --infer_backend vllm \
   --vllm_gpu_memory_utilization 0.8 \
   --vllm_max_model_len 20000 \
   --temperature 1.0 \
   --served_model_name MLLM-4D-SFT-CoT \
   --vllm_limit_mm_per_prompt '{"image": 0, "video": 1}' \
