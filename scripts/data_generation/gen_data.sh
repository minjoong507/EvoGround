GPU_LIST="0,1,2,3"

EXP_ID="EvoGround_Proposer"
PROPOSER_PATH="checkpoints/Proposer/$EXP_ID"
dataset=timer1

export DEBUG_MODE="false" # MUST CHECK THIS
#export DEBUG_MODE="true" # MUST CHECK THIS
export VLLM_WORKER_MULTIPROC_METHOD=spawn

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

for ((i=0; i<num_gpus; i++)); do
    gpu=${gpus[i]}
    PYTHONPATH=$PYTHONPATH:. CUDA_VISIBLE_DEVICES=$gpu python src/gen_data.py \
        --model_base $PROPOSER_PATH \
        --batch_size 8 \
        --curr_idx $i \
        --total_idx $num_gpus \
        --max_new_tokens 512 \
        --split "test" \
        --datasets $dataset \
        --prompt_type "v2" \
        --max_windows 4 \
        --output_dir "gen_data/$EXP_ID/$dataset" \
        --use_vllm_inference &
done
wait

PYTHONPATH=$PYTHONPATH:. python src/postprocess_gen_data.py \
--gen_dataset_path "gen_data/$EXP_ID/$dataset" \
--dataset $dataset
