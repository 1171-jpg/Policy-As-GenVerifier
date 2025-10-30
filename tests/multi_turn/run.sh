CUDA_LAUNCH_BLOCKING=1 TORCH_USE_CUDA_DSA=1 \
MODEL_PATH="Qwen/Qwen2.5-1.5B-Instruct" \
torchrun --standalone --nproc-per-node=1 tests/multi_turn/run_vllm_spmd_pag_rollout.py