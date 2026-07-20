# pip install torch==2.8.0 torchvision==0.23.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cu126
# pip install flash-attn==2.8.0.post2 --no-build-isolation transformers==4.57.1 trl==0.23.0 deepspeed==0.16.8 vllm==0.11.0
# pip install rouge_score wandb decord

output_id=Iter1
OUTDIR="checkpoints/Proposer/Qwen2.5-VL-7B-Proposer-$output_id"
SOLVER_MODEL_NAME_OR_PATH="checkpoints/Qwen-Series/Qwen2.5-VL-7B-Instruct"
BASE_PROPOSER_MODEL_NAME_OR_PATH="checkpoints/Qwen-Series/Qwen2.5-VL-7B-Instruct"

gradient_accumulation_steps=4
dataset="timer1"

export WANDB_PROJECT=Qwen2.5-VL-Proposer
export WANDB_NAME=Iter1

export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="false"
export LOG_PATH="./logs/${WANDB_NAME}_logs.txt"

export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_P2P_DISABLE=1
export NCCL_SHM_DISABLE=1

MASTER_PORT=$(shuf -i 13490-14490 -n 1)

# --bf16 \

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    train_proposer.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_PROPOSER_MODEL_NAME_OR_PATH \
    --solver_model_path $SOLVER_MODEL_NAME_OR_PATH \
    --dataset_name $dataset \
    --max_prompt_length 8192 \
    --max_completion_length 512 \
    --num_generations 2 \
    --per_device_train_batch_size 1 \
    --gradient_accumulation_steps $gradient_accumulation_steps \
    --logging_steps 1 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 1 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --reward_funcs format \
    --temperature 1.0 \
    --prompt_type v2 \
    --max_windows 4 \
    --is_curriculum_learning false \
    --logging_dir $OUTDIR \
    --save_steps 100 \
    --save_only_model true

cd $OUTDIR
python zero_to_fp32.py . .
cd /home/mjjung/workspace/time-r1

export VLLM_WORKER_MULTIPROC_METHOD=spawn

GPU_LIST="0,1,2,3"
IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

for ((i=0; i<num_gpus; i++)); do
    gpu=${gpus[i]}
    PYTHONPATH=$PYTHONPATH:. CUDA_VISIBLE_DEVICES=$gpu python src/gen_data.py \
        --model_base $OUTDIR \
        --batch_size 8 \
        --curr_idx $i \
        --total_idx $num_gpus \
        --max_new_tokens 512 \
        --split "test" \
        --datasets $dataset \
        --prompt_type "v2" \
        --max_windows 4 \
        --output_dir "gen_data/$output_id/timer1" \
        --use_vllm_inference &
done
wait

PYTHONPATH=$PYTHONPATH:. python src/postprocess_gen_data.py \
--org_dataset_path "./dataset/$dataset/annotations/train_2k5.json" \
--gen_dataset_path "gen_data/$output_id/$dataset"