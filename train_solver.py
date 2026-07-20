# Copyright 2025 The HuggingFace Team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import json
import yaml
import math
import os
import random
import re
from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional
from packaging import version

import numpy as np
import torch
from datasets import Dataset
from deepspeed.runtime.fp16.loss_scaler import LossScaler
from deepspeed.runtime.zero.config import ZeroStageEnum
from rouge_score import rouge_scorer
from tqdm import tqdm
import transformers
from transformers import (
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from trl import GRPOConfig, ModelConfig, ScriptArguments, TrlParser, get_peft_config

torch.serialization.add_safe_globals([ZeroStageEnum])
torch.serialization.add_safe_globals([LossScaler])
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

# from src.time_r1 import Solver_Trainer as Trainer
from src.trainer import Solver_Trainer as Trainer

@dataclass
class MY_GRPOConfig(GRPOConfig):
    fix_vit: bool = field(
        default=False,
        metadata={"help": "Whether to fix the ViT model"},
    )

    slide_window: bool = field(
        default=False,
        metadata={"help": "Whether to use slide window"},
    )
    max_window_layers: int = field(
        default=2, metadata={"help": "sliding window layers bottom"}
    )
    sliding_window_length: int = field(
        default=4096, metadata={"help": "sliding window length"}
    )

    prompt_type: str = field(
        default="v1",
        metadata={"help": "Prompt type. Possible values: 'v1', 'v2', 'v3'"},
    )

    use_grpo: bool = field(
        default=False,
        metadata={"help": "Whether to use GRPO"},
    )

    apply_gdpo: bool = field(
        default=False,
        metadata={"help": "Whether to use GDPO"},
    )

    reward_weights: List[float] = field(
        default=None,
        metadata={"help": "Reward weights"},
    )


@dataclass
class GRPOScriptArguments(ScriptArguments):
    """
    Script arguments for the GRPO training script.

    Args:
        reward_funcs (`list[str]`):
            List of reward functions. Possible values: 'iou', 'format'.
    """

    reward_funcs: list[str] = field(
        default_factory=lambda: ["acc", "format"],
        metadata={"help": "List of reward functions. Possible values: 'iou', 'format'"},
    )
    max_pixels: Optional[int] = field(
        default=12845056,
        # default=2809856,
        metadata={"help": "Maximum number of pixels for the image"},
    )
    min_pixels: Optional[int] = field(
        default=3136,
        metadata={"help": "Minimum number of pixels for the image"},
    )

    train_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/train.json",
        metadata={"help": "Path to the training data JSON file."},
    )

    eval_data_path: str = field(
        default="./dataset/finetune/charades/Charades/charades_annotation/val.json",
        metadata={"help": "Path to the evaluation data JSON file."},
    )

    video_folder: str = field(
        default="./dataset/finetune/charades/Charades/Charades_v1",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )

    dataset_info: str = field(
        default="dataset/config.yaml",  # Replace with your actual video folder path
        metadata={"help": "Path to the folder containing video files."},
    )

    is_curriculum_learning: bool = field(
        default=False,
        metadata={"help": "Whether to use curriculum learning."},
    )

    is_early_stopping: bool = field(
        default=False,
        metadata={"help": "Whether to use early stopping"},
    )

def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
    if os.getenv("DEBUG_MODE") == 'true':
        print("last_answer_content:", last_answer_content)

    matches = re.findall(
        r"(\d+\.?\d*) (to|and) (\d+\.?\d*)", last_answer_content, re.IGNORECASE
    )
    if not matches:
        return None
    last_match = matches[-1]
    start_time = float(last_match[0])
    end_time = float(last_match[2])
    return start_time, end_time


def iou_timestamp_reward(
    completions, solution, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    for content, sol in zip(completions, solution):  # Added video_durations

        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union

            reward = iou
        rewards.append(reward)

    return rewards


def accuracy_reward(
    completions, solution, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    durations = kwargs.get("durations")
    for content, sol, duration in zip(
        completions, solution, durations
    ):  # Added video_durations

        reward = 0.0
        parsed_times = parse_timestamp_output(content)
        start_time, end_time = 0, 0
        gt_start, gt_end = sol
        s, e = gt_start, gt_end
        if parsed_times:
            start_time, end_time = parsed_times
            from_number = start_time
            to_number = end_time

            intersection = max(0, min(to_number, e) - max(from_number, s))
            union = max(to_number, e) - min(from_number, s)
            if union > 0:
                iou = intersection / union  # 0.1 0.3
            else:
                iou = 0.0

            gt_start_norm = 1.0 * s / duration
            gt_end_norm = 1.0 * e / duration
            pred_start_norm = 1.0 * start_time / duration
            pred_end_norm = 1.0 * end_time / duration
            reward = (
                iou
                * (1 - abs(gt_start_norm - pred_start_norm))
                * (1 - abs(gt_end_norm - pred_end_norm))
            )

            if os.getenv("DEBUG_MODE") == 'true':
                message = f"------------- {current_time} IoU reward: {reward} -------------\n" + f"Solver Content: {content}" + f"pred second: {str(start_time)}, {str(end_time)}\n" + f"gt second: {str(gt_start)}, {str(gt_end)}\n"
                print(message)

        rewards.append(reward)

    return rewards


def format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
    matches = [re.fullmatch(pattern, content.strip()) for content in completions]
    # print("matches:", matches, completions)
    return [1.0 if match else 0.0 for match in matches]


def extract_think_content(completion: str) -> Optional[str]:
    think_pattern = re.compile(r"<think>(.*?)</think>", re.DOTALL)
    matches = think_pattern.findall(completion)
    if matches:
        return matches[-1].strip()
    return None


reward_funcs_registry = {
    "iou": iou_timestamp_reward,  # Modified registry to use iou_timestamp_reward
    "acc": accuracy_reward,
    "format": format_reward,
}

def load_json_dataset_tg(
    train_data_path, video_folder=None
):

    def create_dataset_from_json(file_path, split_name):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        examples = []

        if os.getenv("DEBUG_MODE") == "true":
            print("---------- DEBUG MODE ----------")
            print("Only take 16 samples.")
            data = data[:16]

        for item in tqdm(data, desc=f"Processing {split_name} items"):
            video_path = item.get("video") if video_folder is None else os.path.join(video_folder, item.get("video"))
            timestamps = item.get("timestamp")
            sentence = item.get("sentence")
            duration = item.get("duration")
            video_start = item.get("video_start")
            video_end = item.get("video_end")

            sentence = sentence.strip().lower()
            if sentence.endswith("."):
                sentence = sentence[:-1]

            if not os.path.isfile(video_path):
                raise FileNotFoundError(f"File {video_path} not found.")
                # continue

            example = {
                "task_type": "tg",
                "problem": sentence,
                "choices": "",
                "solution": (
                    float(timestamps[0]),
                    float(timestamps[1]),
                ),
                "video_path": video_path,
                "durations": duration,
                "video_start": video_start,
                "video_end": video_end,
                "preprocessed_path": "",
            }
            examples.append(example)

        if not examples:
            return None

        random.shuffle(examples)

        for i, ex in enumerate(examples[:5]):
            print(f"  sample: {i+1}: {ex}")

        dataset = Dataset.from_list(examples)
        return dataset

    train_dataset = create_dataset_from_json(train_data_path, "train")

    return train_dataset


class SaveEpochEndCallback(TrainerCallback):
    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.is_world_process_zero:
            trainer = kwargs.get("trainer")
            if trainer is None:
                return

            epoch_checkpoint_dir = os.path.join(
                args.output_dir, f"epoch-{int(state.epoch)}"
            )

            print(
                f"\n{'='*20} Callback: Saving model checkpoint at end of epoch {int(state.epoch)} to {epoch_checkpoint_dir} {'='*20}\n"
            )
            trainer.save_model(epoch_checkpoint_dir)


class StopAfterNEpochsCallback(TrainerCallback):
    def __init__(self, num_epochs_to_train=1):
        super().__init__()
        self.num_epochs_to_train = num_epochs_to_train
        print(
            f"Callback initialized: Training will stop after {self.num_epochs_to_train} completed epoch(s)."
        )

    def on_epoch_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.epoch >= self.num_epochs_to_train:
            print(
                f"Epoch {state.epoch:.0f} completed. Stopping training as per StopAfterNEpochsCallback (target: {self.num_epochs_to_train} epoch(s))."
            )
            control.should_training_stop = True


def set_global_seed(seed_value: int):
    random.seed(seed_value)
    np.random.seed(seed_value)
    torch.manual_seed(seed_value)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed_value)


def main(script_args, training_args, model_args):
    set_global_seed(42)

    # Get reward functions
    reward_funcs = [reward_funcs_registry[func] for func in script_args.reward_funcs]

    # with open(script_args.dataset_info, "r", encoding="utf-8") as f:
    #     dataset_config = yaml.safe_load(f)

    # if script_args.dataset_name in dataset_config:
    #     script_args.video_folder = dataset_config[script_args.dataset_name]["video_folder"]
    #
    # else:
    #     raise FutureWarning(f"Given {script_args.dataset_name} dataset is not registered.")

    dataset = load_json_dataset_tg(
        train_data_path=script_args.train_data_path,
        # video_folder=script_args.video_folder
    )

    trainer_cls = (
        Trainer
    )
    print("using: ", trainer_cls)

    callbacks_list = []
    if script_args.is_early_stopping:
        callbacks_list.append(StopAfterNEpochsCallback())

    # Initialize the GRPO trainer
    trainer = trainer_cls(
        model=model_args.model_name_or_path,
        reward_funcs=reward_funcs,
        args=training_args,
        train_dataset=dataset,
        eval_dataset=None,
        peft_config=get_peft_config(model_args),
        attn_implementation=model_args.attn_implementation,
        max_pixels=script_args.max_pixels,
        min_pixels=script_args.min_pixels,
        callbacks=callbacks_list,
    )

    # Train and push the model to the Hub
    # trainer.train()
    if training_args.resume_from_checkpoint is not None:
        trainer_state_path = os.path.join(
            training_args.resume_from_checkpoint, "trainer_state.json"
        )
        if os.path.exists(trainer_state_path):
            print(f"Loading trainer state from: {trainer_state_path}")
            with open(trainer_state_path, "r") as f:
                trainer_state = json.load(f)
            resumed_global_step = trainer_state.get("global_step", 0)

        num_micro_batches_per_epoch_per_gpu = len(trainer.get_train_dataloader())
        max_step = math.ceil(
            trainer.args.num_train_epochs
            * num_micro_batches_per_epoch_per_gpu
            / trainer.args.gradient_accumulation_steps
        )
        trainer.args.max_steps = resumed_global_step + max_step

        if hasattr(trainer, "state") and hasattr(trainer.state, "max_steps"):
            trainer.state.max_steps = max_step
        else:
            print(
                "Warning: trainer.state.max_steps not found or state not fully initialized. Relying on trainer.args.max_steps."
            )

        print(
            f"Resuming training from checkpoint: {training_args.resume_from_checkpoint}"
        )
        trainer.train(resume_from_checkpoint=training_args.resume_from_checkpoint)
    else:
        trainer.train()

    # Save and push to hub

    if os.getenv("DEBUG_MODE") == "true":
        pass
    else:
        trainer.save_model(training_args.output_dir)
        if training_args.push_to_hub:
            trainer.push_to_hub(dataset_name=script_args.dataset_name)


if __name__ == "__main__":
    parser = TrlParser((GRPOScriptArguments, MY_GRPOConfig, ModelConfig))
    script_args, training_args, model_args = parser.parse_args_and_config()
    main(script_args, training_args, model_args)
