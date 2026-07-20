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

import os
import random
import re
import textwrap
import copy
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
import torch.nn.functional as F
import torch.utils.data
import transformers
from datasets import Dataset, IterableDataset
from packaging import version
from transformers import (
    AutoModelForSequenceClassification,
    AutoProcessor,
    AutoTokenizer,
    GenerationConfig,
    PreTrainedModel,
    PreTrainedTokenizerBase,
    AutoModelForVision2Seq,
    AutoModel,
    Trainer,
    TrainerCallback,
    is_wandb_available,
)
from src.model.model_utils import (
    get_model_type,
    load_model,
    load_processor,
    make_video_content_element,
    call_processor_for_video,
    extract_vision_kwargs,
    repeat_vision_kwargs,
)
from transformers.integrations.deepspeed import is_deepspeed_zero3_enabled
from transformers.utils import is_peft_available
from trl.data_utils import apply_chat_template, is_conversational
from trl.models import (
    create_reference_model,
    prepare_deepspeed,
    unwrap_model_for_generation,
)
from trl.trainer.grpo_config import GRPOConfig
from trl.trainer.utils import generate_model_card, get_comet_experiment_url
from accelerate.utils import gather

import deepspeed
from deepspeed.runtime.zero.stage3 import GatheredParameters

from src.utils import process_vision_info_v3, extract_windows2, DUMMY_WINDOW, PADDING_WINDOW, _read_video_decord_w_timestamp
from src.prompts.prompts import QUESTION_TEMPLATE_TG_v1, SAMPLING_WINDOW_TEMPLATE_v2, SAMPLING_WINDOW_TEMPLATE_v3

prompt_dict = {
    "tg": QUESTION_TEMPLATE_TG_v1,
    "v2": SAMPLING_WINDOW_TEMPLATE_v2,
    "v3": SAMPLING_WINDOW_TEMPLATE_v3,
}

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
MetricFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

import json
import time
from pathlib import Path
import torch.distributed as dist
import numpy as np


def _safe_video_id_from_inputs(example: dict) -> str:
    """
    Robustly derive a stable video_id from common fields.
    Priority: explicit id -> basename(video_path/video) -> 'unknown'
    """

    return example["video_path"].split("/")[-1].split(".")[0]


def _atomic_json_dump(obj, path: str):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with tmp.open("w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)
    os.replace(tmp, path)


def _gather_records_to_rank0(accelerator, local_record: dict):
    """
    Gather arbitrary python objects (dict) from all ranks to rank0.
    Returns list[dict] on rank0, else None.
    """
    # accelerate >= 0.20 usually has gather_object
    if hasattr(accelerator, "gather_object"):
        gathered = accelerator.gather_object(local_record)
        # On main process, gathered is list of records from all ranks
        if accelerator.is_main_process:
            return gathered
        return None

    # fallback
    if dist.is_available() and dist.is_initialized():
        world_size = dist.get_world_size()
        out = [None for _ in range(world_size)]
        dist.all_gather_object(out, local_record)
        if accelerator.is_main_process:
            return out
        return None

    # single process
    if accelerator.is_main_process:
        return [local_record]
    return None


def nanmin(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the minimum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Minimum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.min(tensor[~torch.isnan(tensor)])


def nanmax(tensor: torch.Tensor) -> torch.Tensor:
    """
    Compute the maximum value of a tensor, ignoring NaNs. This function only supports 1D tensors.

    Args:
        tensor (`torch.Tensor`): Input tensor of shape `(N,)`.

    Returns:
        `torch.Tensor`: Maximum value of the tensor, ignoring NaNs. Returns NaN if all values are NaN.
    """
    if torch.isnan(tensor).all():
        return torch.tensor(float("nan"), dtype=tensor.dtype, device=tensor.device)
    return torch.max(tensor[~torch.isnan(tensor)])


class Proposer_Trainer(Trainer):
    """
    Trainer for the Group Relative Policy Optimization (GRPO) method. This algorithm was initially proposed in the
    paper [DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models](https://huggingface.co/papers/2402.03300).

    Example:

    ```python
    from datasets import load_dataset
    from trl import GRPOTrainer

    dataset = load_dataset("trl-lib/tldr", split="train")

    trainer = GRPOTrainer(
        model="Qwen/Qwen2-0.5B-Instruct",
        reward_funcs="weqweasdas/RM-Gemma-2B",
        train_dataset=dataset,
    )

    trainer.train()
    ```

    Args:
        model (`Union[str, PreTrainedModel]`):
            Model to be trained. Can be either:

            - A string, being the *model id* of a pretrained model hosted inside a model repo on huggingface.co, or
              a path to a *directory* containing model weights saved using
              [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is
              loaded using [`~transformers.AutoModelForCausalLM.from_pretrained`] with the keywork arguments
              in `args.model_init_kwargs`.
            - A [`~transformers.PreTrainedModel`] object. Only causal language models are supported.
        reward_funcs (`Union[RewardFunc, list[RewardFunc]]`):
            Reward functions to be used for computing the rewards. To compute the rewards, we call all the reward
            functions with the prompts and completions and sum the rewards. Can be either:

            - A single reward function, such as:
                - A string: The *model ID* of a pretrained model hosted inside a model repo on huggingface.co, or a
                path to a *directory* containing model weights saved using
                [`~transformers.PreTrainedModel.save_pretrained`], e.g., `'./my_model_directory/'`. The model is loaded
                using [`~transformers.AutoModelForSequenceClassification.from_pretrained`] with `num_labels=1` and the
                keyword arguments in `args.model_init_kwargs`.
                - A [`~transformers.PreTrainedModel`] object: Only sequence classification models are supported.
                - A custom reward function: The function is provided with the prompts and the generated completions,
                  plus any additional columns in the dataset. It should return a list of rewards. For more details, see
                  [Using a custom reward function](#using-a-custom-reward-function).
            - A list of reward functions, where each item can independently be any of the above types. Mixing different
            types within the list (e.g., a string model ID and a custom reward function) is allowed.
        args ([`GRPOConfig`], *optional*, defaults to `None`):
            Configuration for this trainer. If `None`, a default configuration is used.
        train_dataset ([`~datasets.Dataset`] or [`~datasets.IterableDataset`]):
            Dataset to use for training. It must include a column `"prompt"`. Any additional columns in the dataset is
            ignored. The format of the samples can be either:

            - [Standard](dataset_formats#standard): Each sample contains plain text.
            - [Conversational](dataset_formats#conversational): Each sample contains structured messages (e.g., role
              and content).
        eval_dataset ([`~datasets.Dataset`], [`~datasets.IterableDataset`] or `dict[str, Union[Dataset, IterableDataset]]`):
            Dataset to use for evaluation. It must meet the same requirements as `train_dataset`.
        processing_class ([`~transformers.PreTrainedTokenizerBase`], *optional*, defaults to `None`):
            Processing class used to process the data. The padding side must be set to "left". If `None`, the
            processing class is loaded from the model's name with [`~transformers.AutoTokenizer.from_pretrained`].
        reward_processing_classes (`Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]`, *optional*, defaults to `None`):
            Processing classes corresponding to the reward functions specified in `reward_funcs`. Can be either:

            - A single processing class: Used when `reward_funcs` contains only one reward function.
            - A list of processing classes: Must match the order and length of the reward functions in `reward_funcs`.
            If set to `None`, or if an element of the list corresponding to a [`~transformers.PreTrainedModel`] is
            `None`, the tokenizer for the model is automatically loaded using [`~transformers.AutoTokenizer.from_pretrained`].
            For elements in `reward_funcs` that are custom reward functions (not [`~transformers.PreTrainedModel`]),
            the corresponding entries in `reward_processing_classes` are ignored.
        callbacks (list of [`~transformers.TrainerCallback`], *optional*, defaults to `None`):
            List of callbacks to customize the training loop. Will add those to the list of default callbacks
            detailed in [here](https://huggingface.co/docs/transformers/main_classes/callback).

            If you want to remove one of the default callbacks used, use the [`~transformers.Trainer.remove_callback`]
            method.
        optimizers (`tuple[torch.optim.Optimizer, torch.optim.lr_scheduler.LambdaLR]`, *optional*, defaults to `(None, None)`):
            A tuple containing the optimizer and the scheduler to use. Will default to an instance of [`AdamW`] on your
            model and a scheduler given by [`get_linear_schedule_with_warmup`] controlled by `args`.
        peft_config ([`~peft.PeftConfig`], *optional*, defaults to `None`):
            PEFT configuration used to wrap the model. If `None`, the model is not wrapped.
    """

    def __init__(
        self,
        model: Union[str, PreTrainedModel],
        reward_funcs: Union[RewardFunc, list[RewardFunc]],
        args: GRPOConfig = None,
        train_dataset: Optional[Union[Dataset, IterableDataset]] = None,
        eval_dataset: Optional[
            Union[Dataset, IterableDataset, dict[str, Union[Dataset, IterableDataset]]]
        ] = None,
        processing_class: Optional[PreTrainedTokenizerBase] = None,
        reward_processing_classes: Optional[
            Union[PreTrainedTokenizerBase, list[PreTrainedTokenizerBase]]
        ] = None,
        callbacks: Optional[list[TrainerCallback]] = None,
        optimizers: tuple[
            Optional[torch.optim.Optimizer], Optional[torch.optim.lr_scheduler.LambdaLR]
        ] = (None, None),
        peft_config: Optional["PeftConfig"] = None,
        max_pixels: Optional[int] = 12845056,
        min_pixels: Optional[int] = 3136,
        attn_implementation: str = "flash_attention_2",
    ):

        # Args
        if args is None:
            model_name = model if isinstance(model, str) else model.config._name_or_path
            model_name = model_name.split("/")[-1]
            args = GRPOConfig(f"{model_name}-GRPO")

        self.args = args
        self.max_windows = args.max_windows
        self.max_windows_feedback = getattr(args, "max_windows_feedback", 8)
        self.consistency_tau = getattr(args, "consistency_tau", 0.07)

        # Models
        model_init_kwargs = args.model_init_kwargs or {}
        model_init_kwargs["attn_implementation"] = attn_implementation

        if isinstance(model, str):
            model_id = model
            torch_dtype = model_init_kwargs.get("torch_dtype")
            if (
                isinstance(torch_dtype, torch.dtype)
                or torch_dtype == "auto"
                or torch_dtype is None
            ):
                pass  # torch_dtype is already a torch.dtype or "auto" or None
            elif isinstance(torch_dtype, str):  # it's a str, but not "auto"
                torch_dtype = getattr(torch, torch_dtype)
                model_init_kwargs["torch_dtype"] = torch_dtype
            else:
                raise ValueError(
                    "Invalid `torch_dtype` passed to `GRPOConfig`. Expected either 'auto' or a string representing "
                    f"a `torch.dtype` (e.g., 'float32'), but got {torch_dtype}."
                )
            # Disable caching if gradient checkpointing is enabled (not supported)
            model_init_kwargs["use_cache"] = (
                False
                if args.gradient_checkpointing
                else model_init_kwargs.get("use_cache")
            )
            print("model_init_kwargs['use_cache']", model_init_kwargs["use_cache"])
            print("model_init_kwargs", model_init_kwargs)
            self.model_init_kwargs = model_init_kwargs

            print(f"Load from {model_id}")
            self.model_type = get_model_type(model_id)
            print(f"[Proposer_Trainer] Detected model_type={self.model_type}")
            model = load_model(
                model_id,
                self.model_type,
                torch_dtype=torch.bfloat16,
                use_sliding_window=getattr(args, "slide_window", False),
                max_window_layers=getattr(args, "max_window_layers", None),
                sliding_window=getattr(args, "sliding_window_length", None),
                use_cache=False,
                attn_implementation="flash_attention_2",
            )

            # print("config", model.config)
        else:
            model_id = model.config._name_or_path
            self.model_type = get_model_type(model_id)
            if args.model_init_kwargs is not None:
                raise ValueError(
                    "You passed `model_init_kwargs` to the `GRPOConfig`, but your model is already instantiated. "
                    "This argument can only be used when the `model` argument is a string."
                )

        if peft_config is not None:
            model = get_peft_model(model, peft_config)

        print("fix_vit", args.fix_vit)
        # Determine if PEFT/LoRA is used
        using_lora = peft_config is not None  # or isinstance(model, PeftModel)
        fix_vit_enabled = hasattr(args, "fix_vit") and args.fix_vit

        if fix_vit_enabled and not using_lora:
            if hasattr(model, "visual"):
                print(
                    "[INFO] fix_vit=True and LoRA not used. Applying ViT freezing logic..."
                )
                model.visual.requires_grad_(False)
                if hasattr(model.visual, "merger"):
                    print("merger exists")
                    model.visual.merger.requires_grad_(True)
            else:
                print(
                    "[WARNING] fix_vit=True but model.visual attribute not found. No freezing applied."
                )
        elif fix_vit_enabled and using_lora:
            print("[INFO] fix_vit=True ignored because LoRA/PEFT is enabled.")
        elif hasattr(args, "fix_vit"):  # fix_vit exists but is False
            print("[INFO] fix_vit=False. ViT freezing logic skipped.")

        self.beta = args.beta
        self.use_grpo = args.use_grpo
        print("self.use_grpo", self.use_grpo)

        if self.beta == 0.0:
            self.ref_model = None
        elif is_deepspeed_zero3_enabled():
            self.ref_model = load_model(
                model_id,
                self.model_type,
                torch_dtype=torch.bfloat16,
                use_sliding_window=getattr(args, "slide_window", False),
                **model_init_kwargs,
            )
        elif peft_config is None:
            self.ref_model = create_reference_model(model)
        else:
            self.ref_model = None

        self.solver = None

        if isinstance(getattr(args, "solver_model_path", None), str):
            self.solver = load_model(
                args.solver_model_path,
                self.model_type,
                torch_dtype=torch.bfloat16,
                use_sliding_window=getattr(args, "slide_window", False),
                **model_init_kwargs,
            )
            print(f"Load the solver model from {args.solver_model_path}")
        else:
            print("Skip loading the solver model.")

        # Processing class
        if processing_class is None:
            processing_class = load_processor(
                model_id, self.model_type, max_pixels=max_pixels, min_pixels=min_pixels
            )
        pad_token_id = processing_class.pad_token_id

        # Reward functions
        if not isinstance(reward_funcs, list):
            reward_funcs = [reward_funcs]
        for i, reward_func in enumerate(reward_funcs):
            if isinstance(reward_func, str):
                print(f"Load model from {reward_func} for the reward function.")
                reward_funcs[i] = AutoModel.from_pretrained(
                    reward_func, torch_dtype=torch.bfloat16,
                )

        self.reward_funcs = reward_funcs

        # Reward weights
        if args.reward_weights is not None:
            if len(args.reward_weights) != len(self.reward_funcs):
                raise ValueError(
                    f"Number of reward weights ({len(args.reward_weights)}) must match number of reward "
                    f"functions ({len(reward_funcs)})"
                )
            self.reward_weights = torch.tensor(args.reward_weights, dtype=torch.float32)
        else:
            print("All reward weights will be 1.0.")
            self.reward_weights = torch.ones(len(self.reward_funcs), dtype=torch.float32)

        if args.apply_gdpo:
            self.apply_gdpo = True
        else:
            self.apply_gdpo = False

        # Reward processing class
        if reward_processing_classes is None:
            reward_processing_classes = [None] * len(reward_funcs)
        elif not isinstance(reward_processing_classes, list):
            reward_processing_classes = [reward_processing_classes]
        else:
            if len(reward_processing_classes) != len(reward_funcs):
                raise ValueError(
                    "The number of reward processing classes must match the number of reward functions."
                )
        for i, (reward_processing_class, reward_func) in enumerate(
            zip(reward_processing_classes, reward_funcs)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if reward_processing_class is None:
                    reward_processing_class = AutoProcessor.from_pretrained(
                        reward_func.config._name_or_path
                    )

                reward_processing_classes[i] = reward_processing_class
        self.reward_processing_classes = reward_processing_classes

        # Data collator
        def data_collator(features):  # No data collation is needed in GRPO
            return features

        # Training arguments
        self.max_prompt_length = args.max_prompt_length
        self.max_completion_length = (
            args.max_completion_length
        )  # = |o_i| in the GRPO paper

        self.num_generations = args.num_generations  # = G in the GRPO paper

        self.temperature = args.temperature
        self.generation_config = GenerationConfig(
            max_new_tokens=self.max_completion_length,
            do_sample=True,
            temperature=self.temperature,  # HACK
            num_return_sequences=self.num_generations,
            pad_token_id=pad_token_id,
        )

        if self.solver is not None:
            self.solver_generation_config = GenerationConfig(
                max_new_tokens=200,
                do_sample=True,
                temperature=self.temperature,  # HACK
                num_return_sequences=1,
                pad_token_id=pad_token_id,
            )

        # self.generation_config.transformers_version = "4.48.3"
        print(
            "self.generation_config.transformers_version",
            self.generation_config.transformers_version,
        )
        args.epsilon = 0.2
        args.epsilon_high = None
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )

        self.prompt_type = args.prompt_type
        self.penalize = True if args.prompt_type in ["v2"] else False

        # The trainer estimates the number of FLOPs (floating-point operations) using the number of elements in the
        # input tensor associated with the key "input_ids". However, in GRPO, the sampled data does not include the
        # "input_ids" key. Instead, the available keys is "prompt". As a result, the trainer issues the warning:
        # "Could not estimate the number of tokens of the input, floating-point operations will not be computed." To
        # suppress this warning, we set the "estimate_tokens" key in the model's "warnings_issued" dictionary to True.
        # This acts as a flag to indicate that the warning has already been issued.
        model.warnings_issued["estimate_tokens"] = True

        # Initialize the metrics
        self._metrics = defaultdict(list)

        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            processing_class=processing_class,
            callbacks=callbacks,
            optimizers=optimizers,
        )

        # Gradient accumulation requires scaled loss. Normally, loss scaling in the parent class depends on whether the
        # model accepts loss-related kwargs. Since we compute our own loss, this check is irrelevant. We set
        # self.model_accepts_loss_kwargs to False to enable scaling.
        self.model_accepts_loss_kwargs = False

        if self.ref_model is not None:
            if self.is_deepspeed_enabled:
                self.ref_model = prepare_deepspeed(self.ref_model, self.accelerator)
            else:
                self.ref_model = self.accelerator.prepare_model(
                    self.ref_model, evaluation_mode=True
                )

        if self.solver is not None:
            if self.is_deepspeed_enabled:
                self.solver = prepare_deepspeed(self.solver, self.accelerator)
            else:
                self.solver = self.accelerator.prepare_model(
                    self.solver, evaluation_mode=True
                )

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                if hasattr(reward_func, "get_image_features"):
                    # CLIP/SigLIP models must NOT go through accelerator.prepare_model:
                    # DeepSpeed ZeRO-3 shards weights into 1D tensors, which breaks
                    # conv2d in patch_embedding (requires ≥3D weight). Move to device directly.
                    self.reward_funcs[i] = reward_func.to(self.accelerator.device).eval()
                else:
                    self.reward_funcs[i] = self.accelerator.prepare_model(
                        reward_func, evaluation_mode=True
                    )

    def _set_signature_columns_if_needed(self):
        # If `self.args.remove_unused_columns` is True, non-signature columns are removed.
        # By default, this method sets `self._signature_columns` to the model's expected inputs.
        # In GRPOTrainer, we preprocess data, so using the model's signature columns doesn't work.
        # Instead, we set them to the columns expected by the `training_step` method, hence the override.
        if self._signature_columns is None:
            self._signature_columns = ["prompt"]

    # Get the per-token log probabilities for the completions for the model and the reference model
    def _get_per_token_logps(
        self, model, input_ids, attention_mask, **vision_kwargs
    ):
        logits = model(
            input_ids,
            attention_mask=attention_mask,
            **vision_kwargs,
        ).logits  # (B, L, V)
        logits = logits[
            :, :-1, :
        ]  # (B, L-1, V), exclude the last logit: it corresponds to the next token pred
        input_ids = input_ids[
            :, 1:
        ]  # (B, L-1), exclude the first input ID since we don't have logits for it
        # Compute the log probabilities for the input tokens. Use a loop to reduce memory peak.
        per_token_logps = []
        per_token_entropy = []

        for logits_row, input_ids_row in zip(logits, input_ids):
            log_probs = logits_row.log_softmax(dim=-1)
            token_log_prob = torch.gather(
                log_probs, dim=1, index=input_ids_row.unsqueeze(1)
            ).squeeze(1)
            per_token_logps.append(token_log_prob)

            # calc H(p) = - sum(p * log p)
            # p = exp(log_probs)
            probs = torch.exp(log_probs)  # (L-1, V)
            entropy = -torch.sum(probs * log_probs, dim=-1)  # (L-1)
            per_token_entropy.append(entropy)

        return torch.stack(per_token_logps), torch.stack(per_token_entropy)

    # Trainer "prepares" the inputs before calling `compute_loss`. It converts to tensor and move to device.
    # Since we preprocess the data in `compute_loss`, we need to override this method to skip this step.
    def _prepare_inputs(
        self, inputs: dict[str, Union[torch.Tensor, Any]]
    ) -> dict[str, Union[torch.Tensor, Any]]:
        return inputs

    def make_conversation_video(self, example, prompt_type="v2", caption=None):
        _NUM_TO_WORD = {
            1: "one", 2: "two", 3: "three", 4: "four", 5: "five",
            6: "six", 7: "seven", 8: "eight", 9: "nine", 10: "ten",
        }

        def _num_to_word(n: int) -> str:
            return _NUM_TO_WORD.get(n, str(n))  # fallback to digit string if out of range

        template = prompt_dict[prompt_type]
        if prompt_type in ["tg"]:
            if caption is None:
                raise NotImplementedError("Check the caption input")
            prompt_text = template.replace("[EVENT]", str(caption))

        elif prompt_type in ["v2", "v3"]:
            prompt_text = template.replace("[DURATION]", str(example["durations"]))
            if "[MAX_WINDOW]" in prompt_text:
                prompt_text = prompt_text.replace("[MAX_WINDOW]", _num_to_word(self.max_windows))

        else:
            raise NotImplementedError

        if os.getenv("DEBUG_MODE") == "true":
            print(f"{[prompt_type]} Question prompt", prompt_text)

        video_ele = make_video_content_element(
            self.model_type,
            example["video_path"],
            video_start=example["video_start"],
            video_end=example["video_end"],
        )

        return [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt_text},
                        video_ele,
                    ],
                },
            ]

    def compute_consistency_rewards(
        self,
        model: PreTrainedModel,
        processing_class,
        input: dict,
        windows_set: list,
        captions_set: list,
        device,
        fps_rate: float = 1.0,
        min_frames: int = 8,
        max_frames: int = 32,
        gamma: float = 30.0,
    ) -> list[float]:
        """
        Compute per-generation consistency reward scores using adaptive frame sampling.

        For each generation (windows, captions):
          1. Sample T_i = clip(duration_i * fps_rate, min_frames, max_frames) frames per window.
          2. Extract per-frame visual features via model.get_image_features → list of (T_i, D).
          3. Extract text features via model.get_text_features → (N, D).
          4. Compute inter-consistency via softmax(mu_raw / τ) diagonal.
          5. Compute intra-consistency = exp(-gamma * rms_residual) per (moment, query).
          6. reward = sum(inter * intra diagonal) / total_non_padding.

        Returns:
            rewards: list[float] – one scalar per generation
        """

        try:
            video_path  = input["video_path"]
            video_start = float(input["video_start"]) if isinstance(input["video_start"], (int, float)) else 0.0
            video_end   = float(input["video_end"])   if isinstance(input["video_end"],   (int, float)) else float(input["durations"])
            ele = {"video": video_path, "video_start": video_start, "video_end": video_end}
        except KeyError as e:
            print(f"[consistency reward] Missing input key: {e}")
            return [0.0] * len(windows_set)

        try:
            # vr: (T_seg, C, H, W) tensor covering [video_start, video_end] only
            # fps: effective sample fps of the returned tensor
            vr, fps = _read_video_decord_w_timestamp(ele)
        except Exception as e:
            print(f"[consistency reward] Failed to open video {video_path}: {e}")
            return [0.0] * len(windows_set)

        total_frames = len(vr)  # number of frames in the sliced segment

        rewards: list[float] = []

        with torch.inference_mode(), GatheredParameters(
            list(model.parameters()), enabled=is_deepspeed_zero3_enabled()
        ):
            for windows, captions in zip(windows_set, captions_set):
                non_padding_indices = [i for i, w in enumerate(windows) if w != PADDING_WINDOW]
                valid_indices       = [i for i in non_padding_indices if windows[i] != DUMMY_WINDOW]
                total_non_padding   = len(non_padding_indices)

                if not valid_indices:
                    rewards.append(0.0)
                    continue

                valid_windows  = [windows[i]  for i in valid_indices]
                valid_captions = [captions[i] for i in valid_indices]
                N = len(valid_windows)

                # --- Visual features: list of N tensors, each (T_i, D) ---
                seg_feats_list: list[torch.Tensor] = []
                for start, end in valid_windows:
                    # Offset to segment-relative time, then convert to frame indices
                    rel_start   = max(start - video_start, 0.0)
                    rel_end     = min(end   - video_start, total_frames / fps)
                    T_i         = int(np.clip((rel_end - rel_start) * fps_rate, min_frames, max_frames))
                    start_frame = int(rel_start * fps)
                    end_frame   = max(int(rel_end * fps), start_frame + 1)
                    indices     = np.linspace(start_frame, end_frame, T_i).astype(int)
                    indices     = np.clip(indices, 0, total_frames - 1)
                    # vr is (T_seg, C, H, W); convert sampled frames to (T_i, H, W, C) for processor
                    frames      = vr[indices].permute(0, 2, 3, 1).numpy()  # (T_i, H, W, C)

                    visual_inputs = processing_class(
                        images=list(frames), return_tensors="pt", padding=True
                    ).to(device)
                    feats = model.get_image_features(**visual_inputs)   # (T_i, D)
                    feats = feats / feats.norm(dim=-1, keepdim=True)
                    seg_feats_list.append(feats)                        # (T_i, D)

                # --- Text features (N, D) ---
                text_inputs = processing_class(
                    text=valid_captions, return_tensors="pt",
                    padding=True, truncation=True, max_length=64,
                ).to(device)
                text_feats = model.get_text_features(**text_inputs)     # (N, D)
                text_feats = text_feats / text_feats.norm(dim=-1, keepdim=True)

                # --- consistency matrix ---
                # Pass 1: per-frame sim vectors, μ_matched (diagonal), and μ_raw (full).
                sim_rows_local: list[torch.Tensor] = []
                mu_rows_all:    list[torch.Tensor] = []
                mu_matched_list: list[torch.Tensor] = []
                for m, f_m in enumerate(seg_feats_list):                # f_m: (T_m, D)
                    sim_row = f_m @ text_feats.T                        # (T_m, N)
                    sim_rows_local.append(sim_row)
                    mu_row = sim_row.mean(dim=0)                        # (N,)
                    mu_rows_all.append(mu_row)
                    mu_matched_list.append(mu_row[m])                   # scalar: matched mean

                mu_matched = torch.stack(mu_matched_list).float()       # (N,)
                mu_raw     = torch.stack(mu_rows_all).float()           # (M, N)

                # Pass 2: consistency[m, n] = exp(-γ · std)
                consistency_rows: list[torch.Tensor] = []
                for m, sim_row in enumerate(sim_rows_local):            # sim_row: (T_m, N)
                    residuals = sim_row - mu_matched[m]                 # (T_m, N)
                    rms_row = residuals.pow(2).mean(dim=0).sqrt()       # (N,)
                    consistency_rows.append(torch.exp(-gamma * rms_row))       # (N,)

                consistency_matrix = torch.stack(consistency_rows)               # (M, N)

                inter_consistency = torch.diagonal(
                    F.softmax(mu_raw / self.consistency_tau, dim=0)
                ).float()

                # Final reward = inter-consistency × intra-consistency (diagonal)
                intra_consistency = torch.diagonal(consistency_matrix)       # (N,)
                diag_rewards   = inter_consistency * intra_consistency       # (N,)

                if os.getenv("DEBUG_MODE") == "true":
                    for window, caption, diag_rewad, in zip(windows_set, captions_set, diag_rewards):
                        print("\n window: ", window, " caption: ", caption, " consistency: ", diag_rewad.item())

                reward = diag_rewards.sum().item() / max(total_non_padding, 1)

                rewards.append(reward)

        return rewards

    def compute_loss(
            self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        if inputs[0]["use_preprocessed"]:
            video_inputs = [x["video_inputs"] for x in inputs]
            fps_inputs = [x["video_kwargs"]["fps"] for x in inputs]
            video_inputs = video_inputs[0]
            fps_inputs = fps_inputs[0]
        else:
            image_inputs, video_inputs, video_kwargs = process_vision_info_v3(
                self.make_conversation_video(inputs[0], prompt_type=self.prompt_type), return_video_kwargs=True
            )
            fps_inputs = video_kwargs["fps"]

        # -------------------------------------------------------------------------
        # 1. Temporal window proposal
        # -------------------------------------------------------------------------
        prompts = [self.make_conversation_video(example, prompt_type=self.prompt_type) for example in inputs]
        prompts_text = [
            self.processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        prompt_inputs = call_processor_for_video(
            self.processing_class,
            self.model_type,
            [prompts_text[0]],
            [video_inputs[0]],
            [fps_inputs[0]],
            padding=True,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
        )
        prompt_inputs = super()._prepare_inputs(prompt_inputs)
        prompt_ids, prompt_mask = prompt_inputs["input_ids"], prompt_inputs["attention_mask"]

        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
            )
            prompt_length = prompt_ids.size(1)
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )

        # -------------------------------------------------------------------------
        # 2. Phase 1 — Extract all windows and captions upfront
        # -------------------------------------------------------------------------
        windows_set = []
        captions_set = []
        solver_completions = []

        for completion in completions:
            windows, captions = extract_windows2(
                completion,
                inputs[0]["durations"],
                max_window=self.max_windows,
                total_window=self.max_windows_feedback,
                min_length=3,
            )

            windows_set.append(windows)
            captions_set.append(captions)

            if self.solver is not None:
                with torch.inference_mode():
                    for i, captions in enumerate(captions_set):
                        # Build all window prompts for this generation at once
                        solver_prompts_text = [
                            self.processing_class.apply_chat_template(
                                self.make_conversation_video(inputs[0], prompt_type="tg", caption=caption),
                                tokenize=False,
                                add_generation_prompt=True,
                            )
                            for caption in captions
                        ]
                        n = len(solver_prompts_text)

                        solver_inputs = call_processor_for_video(
                            self.processing_class,
                            self.model_type,
                            solver_prompts_text,
                            [video_inputs[0]] * n,
                            [fps_inputs[0]] * n,
                            padding=True,
                            return_tensors="pt",
                            padding_side="left",
                            add_special_tokens=False,
                        )
                        solver_inputs = super()._prepare_inputs(solver_inputs)
                        solver_prompt_length = solver_inputs["input_ids"].size(1)

                        with unwrap_model_for_generation(self.solver, self.accelerator) as unwrapped_solver:
                            solver_out_ids = unwrapped_solver.generate(
                                **solver_inputs,
                                generation_config=self.solver_generation_config,
                                use_model_defaults=False,
                            )

                        decoded = self.processing_class.batch_decode(
                            solver_out_ids[:, solver_prompt_length:],
                            skip_special_tokens=True,
                        )
                        solver_completions.append(decoded)

        # -------------------------------------------------------------------------
        # 4. Mask everything after the first EOS token
        # -------------------------------------------------------------------------
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full((is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device)
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(is_eos.size(0), -1)
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)
        vision_kwargs = extract_vision_kwargs(self.model_type, prompt_inputs)
        repeated_vision_kwargs = repeat_vision_kwargs(
            self.model_type, vision_kwargs, self.num_generations
        )

        per_token_logps, per_token_entropy = self._get_per_token_logps(
            model, prompt_completion_ids, attention_mask, **repeated_vision_kwargs,
        )
        per_token_logps = per_token_logps[:, prompt_length - 1:]
        entropy_completion = per_token_entropy[:, prompt_length - 1:]

        # -------------------------------------------------------------------------
        # 5. Reference model KL
        # -------------------------------------------------------------------------
        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps(
                        self.ref_model, prompt_completion_ids, attention_mask,
                        **repeated_vision_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps(
                            model, prompt_completion_ids, attention_mask,
                            **repeated_vision_kwargs,
                        )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1:]
            per_token_kl = (
                    torch.exp(ref_per_token_logps - per_token_logps)
                    - (ref_per_token_logps - per_token_logps)
                    - 1
            )

        # -------------------------------------------------------------------------
        # 6. Rewards
        # -------------------------------------------------------------------------
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": c}] for c in completions
            ]

        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]
        rewards_per_func = torch.zeros(len(prompts), len(self.reward_funcs), device=device)

        for i, (reward_func, reward_processing_class) in enumerate(
                zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                consistency_rewards = self.compute_consistency_rewards(
                    reward_func, reward_processing_class, inputs[0], windows_set, captions_set, device
                )
                rewards_per_func[:, i] = torch.tensor(consistency_rewards, dtype=torch.float32, device=device)

            else:
                reward_kwargs = {
                    key: [] for key in inputs[0].keys() if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)

                if hasattr(self.args, "format_iou_thd"):
                    reward_kwargs["format_iou_thd"] = self.args.format_iou_thd

                output_reward_func = reward_func(
                    prompts=prompts,
                    completions=completions,
                    solver_completions=solver_completions,
                    windows_set=windows_set,
                    **reward_kwargs,
                )

                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # -------------------------------------------------------------------------
        # 7. Advantage computation (GDPO or standard GRPO)
        # -------------------------------------------------------------------------
        if self.apply_gdpo and len(self.reward_weights) > 1:
            rewards_per_func_filter = torch.nan_to_num(rewards_per_func)
            all_reward_advantage = []

            for i in range(len(self.reward_weights)):
                reward_i = rewards_per_func_filter[:, i]
                mean_g = reward_i.view(-1, self.num_generations).mean(dim=1).repeat_interleave(self.num_generations,
                                                                                               dim=0)
                std_g = reward_i.view(-1, self.num_generations).std(dim=1).repeat_interleave(self.num_generations,
                                                                                             dim=0)
                all_reward_advantage.append((reward_i - mean_g) / (std_g + 1e-4))

            combined = torch.stack(all_reward_advantage, dim=1)
            pre_bn = (combined * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            advantages = (pre_bn - pre_bn.mean()) / (pre_bn.std() + 1e-4)

        else:
            rewards = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
            mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
            std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
            mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            std_grouped_rewards = std_grouped_rewards.repeat_interleave(self.num_generations, dim=0)
            advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)

        # -------------------------------------------------------------------------
        # 8. Policy loss
        # -------------------------------------------------------------------------
        if self.use_grpo:
            per_token_loss = torch.exp(per_token_logps - per_token_logps.detach()) * advantages.unsqueeze(1)
            per_token_loss = -(per_token_loss - self.beta * per_token_kl) if self.beta != 0.0 else -per_token_loss
            loss = ((per_token_loss * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
        else:
            coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss = -torch.min(coef_1 * advantages.unsqueeze(1), coef_2 * advantages.unsqueeze(1))
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # -------------------------------------------------------------------------
        # 9. Metrics logging
        # -------------------------------------------------------------------------
        self._metrics["completion_length"].append(
            self.accelerator.gather_for_metrics(completion_mask.sum(1)).float().mean().item()
        )

        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            name = (
                reward_func.config._name_or_path.split("/")[-1]
                if isinstance(reward_func, PreTrainedModel)
                else reward_func.__name__
            )
            if "siglip" in name:
                name = "consistency"
            self._metrics[f"rewards/{name}"].append(reward_per_func[i].item())

        # Reward mean/std reporting (shared path for both GDPO and standard)
        rewards_for_log = (rewards_per_func * self.reward_weights.to(device).unsqueeze(0)).nansum(dim=1)
        mean_grouped_rewards_log = rewards_for_log.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards_log = rewards_for_log.view(-1, self.num_generations).std(dim=1)
        self._metrics["reward"].append(mean_grouped_rewards_log.detach().mean().item())
        self._metrics["reward_std"].append(std_grouped_rewards_log.detach().mean().item())

        if self.beta != 0.0:
            mean_kl = ((per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)).mean()
            self._metrics["kl"].append(self.accelerator.gather_for_metrics(mean_kl).mean().item())

        completion_lengths = completion_mask.sum(dim=1).clamp(min=1)
        batch_mean_entropy = (entropy_completion * completion_mask).sum(dim=1).div(completion_lengths).mean()
        self._metrics["generation_entropy"].append(
            self.accelerator.gather_for_metrics(batch_mean_entropy).mean().item()
        )

        all_captions = [caption for captions in captions_set for caption in captions]
        avg_caption_length = sum(len(c) for c in all_captions) / len(all_captions) if all_captions else 0.0
        self._metrics["caption/avg_char_length"].append(avg_caption_length)

        # Average word count per generated caption
        avg_caption_word_count = sum(len(c.split()) for c in all_captions) / len(all_captions) if all_captions else 0.0
        self._metrics["caption/avg_word_count"].append(avg_caption_word_count)

        # Average moment duration in seconds: window = [start, end]
        all_windows = [window for windows in windows_set for window in windows]
        avg_moment_duration = sum(w[1] - w[0] for w in all_windows) / len(all_windows) if all_windows else 0.0
        self._metrics["moment/avg_duration_sec"].append(avg_moment_duration)

        # Average moment duration as a fraction of the total video duration
        total_duration = inputs[0]["durations"]
        avg_moment_duration_ratio = avg_moment_duration / total_duration if total_duration > 0 else 0.0
        self._metrics["moment/avg_duration_ratio"].append(avg_moment_duration_ratio)

        if torch.cuda.is_available():
            import gc
            gc.collect()
            torch.cuda.empty_cache()

        return loss

    def log(self, logs: dict[str, float], start_time: Optional[float] = None) -> None:
        metrics = {
            key: sum(val) / len(val) for key, val in self._metrics.items()
        }  # average the metrics
        logs = {**logs, **metrics}
        if version.parse(transformers.__version__) >= version.parse("4.47.0.dev0"):
            super().log(logs, start_time)
        else:  # transformers<=4.46
            super().log(logs)
        self._metrics.clear()

    def save_model(self, output_dir: Optional[str] = None, _internal_call: bool = False):
        if output_dir is None:
            output_dir = self.args.output_dir

        if self.is_deepspeed_enabled:
            # 1. Clear memory before the save starts
            import gc
            gc.collect()
            torch.cuda.empty_cache()

            # 2. Bypass the materialization spike and go straight to sharded saving
            # This writes shards from GPU to Disk, skipping the CPU RAM bottleneck
            self.model_wrapped.save_checkpoint(output_dir)

            if self.args.should_save:
                # Save metadata (config/processor) without weights to avoid OOM
                self._save(output_dir, state_dict={})
        else:
            super().save_model(output_dir, _internal_call)

    def create_model_card(
        self,
        model_name: Optional[str] = None,
        dataset_name: Optional[str] = None,
        tags: Union[str, list[str], None] = None,
    ):
        """
        Creates a draft of a model card using the information available to the `Trainer`.

        Args:
            model_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the model.
            dataset_name (`str` or `None`, *optional*, defaults to `None`):
                Name of the dataset used for training.
            tags (`str`, `list[str]` or `None`, *optional*, defaults to `None`):
                Tags to be associated with the model card.
        """
        if not self.is_world_process_zero():
            return

        if hasattr(self.model.config, "_name_or_path") and not os.path.isdir(
            self.model.config._name_or_path
        ):
            base_model = self.model.config._name_or_path
        else:
            base_model = None

        tags = tags or []
        if isinstance(tags, str):
            tags = [tags]

        if hasattr(self.model.config, "unsloth_version"):
            tags.append("unsloth")

        citation = textwrap.dedent(
            """\
            @article{zhihong2024deepseekmath,
                title        = {{DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models}},
                author       = {Zhihong Shao and Peiyi Wang and Qihao Zhu and Runxin Xu and Junxiao Song and Mingchuan Zhang and Y. K. Li and Y. Wu and Daya Guo},
                year         = 2024,
                eprint       = {arXiv:2402.03300},
            """
        )

        model_card = generate_model_card(
            base_model=base_model,
            model_name=model_name,
            hub_model_id=self.hub_model_id,
            dataset_name=dataset_name,
            tags=tags,
            wandb_url=(
                wandb.run.get_url()
                if is_wandb_available() and wandb.run is not None
                else None
            ),
            comet_url=get_comet_experiment_url(),
            trainer_name="GRPO",
            trainer_citation=citation,
            paper_title="DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models",
            paper_id="2402.03300",
        )

        model_card.save(os.path.join(self.args.output_dir, "README.md"))
