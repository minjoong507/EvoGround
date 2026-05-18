#!/bin/bash
# testing MODEL_NAME on $EVAL_DATASET dataset using vLLM inference
# $EVAL_DATASET filepath: ./dataset/$EVAL_DATASET

export VLLM_WORKER_MULTIPROC_METHOD=spawn

GPU_LIST="0,1,2,3"

EXP_ID="EvoGround"
BASE_PATH="checkpoints/EvoGround"

export DEBUG_MODE="false"
# specify the dataset you want to use, choose from: ["charades", "activitynet", "tvgbench", "videomme", "tempcompass", "rextimeTG", "etbenchTG]

IFS=',' read -ra gpus <<< "$GPU_LIST"
num_gpus=${#gpus[@]}

for eval_dataset in tvgbench; do
  echo "===============================Running eval on $eval_dataset ==============================="
  for ((i=0; i<num_gpus; i++)); do
      gpu=${gpus[i]}
      PYTHONPATH=$PYTHONPATH:. CUDA_VISIBLE_DEVICES=$gpu python evaluate.py \
          --model_base $BASE_PATH \
          --batch_size 8 \
          --curr_idx $i \
          --total_idx $num_gpus \
          --max_new_tokens 200 \
          --split "test" \
          --datasets $eval_dataset \
          --output_dir "outputs/eval/$EXP_ID/$eval_dataset" \
          --use_vllm_inference \
          --use_r1_thinking_prompt \
          --use_nothink & # uncomment this line to use no-think prompt, especially for VQA tasks
  done
  wait

  echo "===============================Results on $eval_dataset ==============================="
  PYTHONPATH=$PYTHONPATH:. python src/vllm_inference/eval_all.py --model_name $BASE_PATH --split "test" --dataset $eval_dataset --exp_id $EXP_ID
done
