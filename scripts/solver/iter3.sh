OUTDIR="checkpoints/Solver/Qwen2.5-VL-7B-Solver-Iter3"
GEN_DATA_SOURCE="Qwen2.5-VL-7B-Proposer-Iter2"
BASE_MODEL_NAME_OR_PATH="checkpoints/Solver/Qwen2.5-VL-7B-Solver-Iter2"

export WANDB_PROJECT=Qwen2.5-VL-Solver
export WANDB_NAME=Iter3

export PYTHONPATH=".:$PYTHONPATH"
export DEBUG_MODE="false"
export RUN_EVAL="true"
export LOG_PATH="./logs/${WANDB_NAME}_logs.txt"

MASTER_PORT=$(shuf -i 13490-14490 -n 1)

torchrun --nproc_per_node="4" \
    --nnodes="1" \
    --node_rank="0" \
    --master_addr="127.0.0.1" \
    --master_port=$MASTER_PORT \
    train_solver.py \
    --deepspeed scripts/zero3_offload.json \
    --output_dir $OUTDIR \
    --model_name_or_path $BASE_MODEL_NAME_OR_PATH \
    --train_data_path ./gen_data/$GEN_DATA_SOURCE/timer1/gen_data.json \
    --dataset_name xxx \
    --max_prompt_length 8192 \
    --max_completion_length 200 \
    --num_generations 2 \
    --per_device_train_batch_size 2 \
    --gradient_accumulation_steps 4 \
    --logging_steps 1 \
    --torch_dtype bfloat16 \
    --data_seed 42 \
    --gradient_checkpointing true \
    --attn_implementation flash_attention_2 \
    --fix_vit true \
    --slide_window false \
    --num_train_epochs 3 \
    --run_name $WANDB_NAME \
    --report_to wandb \
    --reward_funcs acc format \
    --temperature 1.0 \
    --prompt_type v1 \
    --is_curriculum_learning false \
    --logging_dir $BASE_MODEL_NAME_OR_PATH \
    --save_steps 200 \
    --save_only_model true


 ── Evaluation ────────────────────────────────────────────────────────────────
if [ "RUN_EVAL" = "true" ]; then
  export VLLM_WORKER_MULTIPROC_METHOD=spawn

  GPU_LIST="0, 1, 2, 3"
  IFS=',' read -ra gpus <<< "$GPU_LIST"
  num_gpus=${#gpus[@]}

  EXP_ID=$(basename $OUTDIR)

  for eval_dataset in tvgbench; do
    echo "===============================Running eval on $eval_dataset ==============================="
    for ((i=0; i<num_gpus; i++)); do
        gpu=${gpus[i]}
        PYTHONPATH=$PYTHONPATH:. CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
            --model_base $OUTDIR \
            --batch_size 8 \
            --curr_idx $i \
            --total_idx $num_gpus \
            --max_new_tokens 200 \
            --split "test" \
            --datasets $eval_dataset \
            --output_dir "outputs/eval/$EXP_ID/$eval_dataset" \
            --use_vllm_inference \
            --use_r1_thinking_prompt \
            --use_nothink &
    done
    wait

    echo "===============================Results on $eval_dataset ==============================="
    PYTHONPATH=$PYTHONPATH:. python src/vllm_inference/eval_all.py --model_name $OUTDIR --split "test" --dataset $eval_dataset --exp_id $EXP_ID
  done
fi