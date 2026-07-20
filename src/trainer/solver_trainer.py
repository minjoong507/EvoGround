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
import textwrap
from collections import defaultdict
from typing import Any, Callable, Optional, Union

import torch
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

from src.utils import process_vision_info_v3

if is_peft_available():
    from peft import PeftConfig, get_peft_model

if is_wandb_available():
    import wandb

# What we call a reward function is a callable that takes a list of prompts and completions and returns a list of
# rewards. When it's a string, it's a model ID, so it's loaded as a pretrained model.
RewardFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]
MetricFunc = Union[str, PreTrainedModel, Callable[[list, list], list[float]]]

from src.prompts.prompts import QUESTION_TEMPLATE_TG_v1

os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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


class Solver_Trainer(Trainer):
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

        # Models
        # Trained model
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
            # model_init_kwargs["use_cache"] = (
            #     False
            #     if args.gradient_checkpointing
            #     else model_init_kwargs.get("use_cache")
            # )
            # print("model_init_kwargs['use_cache']", model_init_kwargs["use_cache"])
            # print("model_init_kwargs", model_init_kwargs)
            self.model_type = get_model_type(model_id)
            print(f"[Solver_Trainer] Detected model_type={self.model_type}")
            model = load_model(
                model_id,
                self.model_type,
                torch_dtype=torch.bfloat16,
                use_sliding_window=getattr(args, "slide_window", False),
                max_window_layers=getattr(args, "max_window_layers", None),
                sliding_window=getattr(args, "sliding_window_length", None),
                **model_init_kwargs,
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

        print("model.config.transformers_version", model.config.transformers_version)
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
        # Reference model

        self.beta = args.beta
        self.use_grpo = args.use_grpo

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
                reward_funcs[i] = AutoModelForSequenceClassification.from_pretrained(
                    reward_func, num_labels=1, **model_init_kwargs
                )
        self.reward_funcs = reward_funcs

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
                    reward_processing_class = AutoTokenizer.from_pretrained(
                        reward_func.config._name_or_path
                    )
                if reward_processing_class.pad_token_id is None:
                    reward_processing_class.pad_token = (
                        reward_processing_class.eos_token
                    )
                reward_func.config.pad_token_id = reward_processing_class.pad_token_id
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

        args.epsilon = 0.2
        args.epsilon_high = None
        self.epsilon_low = args.epsilon
        self.epsilon_high = (
            args.epsilon_high if args.epsilon_high is not None else args.epsilon
        )

        self.prompt_type = args.prompt_type

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

        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
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

    def make_conversation_video(self, example):
        prompt_text = QUESTION_TEMPLATE_TG_v1.replace("[EVENT]", example["problem"])

        if os.getenv("DEBUG_MODE") == "true":
            print("question prompt", prompt_text)

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

    def compute_loss(
        self, model, inputs, return_outputs=False, num_items_in_batch=None
    ):
        if return_outputs:
            raise ValueError("The GRPOTrainer does not support returning outputs")

        # ── Load video for every example in the batch ──────────────────────
        # Build one minimal message per example so that the vision-info helper
        # can handle temporal trimming (video_start / video_end) for each video.
        all_video_messages = [
            [{"role": "user", "content": [make_video_content_element(
                self.model_type,
                ex["video_path"],
                video_start=ex.get("video_start"),
                video_end=ex.get("video_end"),
            )]}]
            for ex in inputs
        ]

        _, video_inputs, video_kwargs = process_vision_info_v3(
            all_video_messages, return_video_kwargs=True
        )
        fps_inputs = video_kwargs["fps"]

        # ── Build prompts for all B examples ───────────────────────────────
        prompts = [self.make_conversation_video(example) for example in inputs]

        prompts_text = [
            self.processing_class.apply_chat_template(
                prompt, tokenize=False, add_generation_prompt=True
            )
            for prompt in prompts
        ]

        # Process all B (text, video) tuples at once.
        # The processor left-pads all prompts to the same length (P_max).
        prompt_inputs = call_processor_for_video(
            self.processing_class,
            self.model_type,
            prompts_text,       # B texts
            video_inputs,       # B video tensors
            fps_inputs,
            padding=True,
            return_tensors="pt",
            padding_side="left",
            add_special_tokens=False,
        )

        prompt_inputs = super()._prepare_inputs(prompt_inputs)

        prompt_ids, prompt_mask = (
            prompt_inputs["input_ids"],
            prompt_inputs["attention_mask"],
        )
        vision_kwargs = extract_vision_kwargs(self.model_type, prompt_inputs)

        # Generate completions
        with unwrap_model_for_generation(model, self.accelerator) as unwrapped_model:
            prompt_completion_ids = unwrapped_model.generate(
                **prompt_inputs,
                generation_config=self.generation_config,
                use_model_defaults=False,
            )

            prompt_length = prompt_ids.size(1)
            prompt_ids = prompt_completion_ids[:, :prompt_length]
            completion_ids = prompt_completion_ids[:, prompt_length:]
            prompt_mask = prompt_mask.repeat_interleave(self.num_generations, dim=0)

        # Mask everything after the first EOS token
        is_eos = completion_ids == self.processing_class.eos_token_id
        device = self.accelerator.device
        eos_idx = torch.full(
            (is_eos.size(0),), is_eos.size(1), dtype=torch.long, device=device
        )
        eos_idx[is_eos.any(dim=1)] = is_eos.int().argmax(dim=1)[is_eos.any(dim=1)]
        sequence_indices = torch.arange(is_eos.size(1), device=device).expand(
            is_eos.size(0), -1
        )
        completion_mask = (sequence_indices <= eos_idx.unsqueeze(1)).int()

        # Concatenate prompt_mask with completion_mask for logit computation
        attention_mask = torch.cat([prompt_mask, completion_mask], dim=1)  # (B*G, P+C)
        repeated_vision_kwargs = repeat_vision_kwargs(
            self.model_type, vision_kwargs, self.num_generations
        )

        per_token_logps, per_token_entropy = self._get_per_token_logps(
            model,
            prompt_completion_ids,
            attention_mask,
            **repeated_vision_kwargs,
        )
        # Get rid of the prompt (-1 because of the shift done in get_per_token_logps)
        per_token_logps = per_token_logps[:, prompt_length - 1 :]

        # logps_completion = per_token_logps[:, prompt_length - 1 :]
        entropy_completion = per_token_entropy[:, prompt_length - 1 :]
        if self.beta != 0.0:
            with torch.inference_mode():
                if self.ref_model is not None:
                    ref_per_token_logps, _ = self._get_per_token_logps(
                        self.ref_model,
                        prompt_completion_ids,
                        attention_mask,
                        **repeated_vision_kwargs,
                    )
                else:
                    with self.accelerator.unwrap_model(model).disable_adapter():
                        ref_per_token_logps, _ = self._get_per_token_logps(
                            model,
                            prompt_completion_ids,
                            attention_mask,
                            **repeated_vision_kwargs,
                        )
            ref_per_token_logps = ref_per_token_logps[:, prompt_length - 1 :]

            # Compute the KL divergence between the model and the reference model
            per_token_kl = (
                torch.exp(ref_per_token_logps - per_token_logps)
                - (ref_per_token_logps - per_token_logps)
                - 1
            )

        # Decode the generated completions
        completions = self.processing_class.batch_decode(
            completion_ids, skip_special_tokens=True
        )
        if is_conversational(inputs[0]):
            completions = [
                [{"role": "assistant", "content": completion}]
                for completion in completions
            ]

        # Compute the rewards
        prompts = [prompt for prompt in prompts for _ in range(self.num_generations)]

        rewards_per_func = torch.zeros(
            len(prompts), len(self.reward_funcs), device=device
        )

        for i, (reward_func, reward_processing_class) in enumerate(
            zip(self.reward_funcs, self.reward_processing_classes)
        ):
            if isinstance(reward_func, PreTrainedModel):
                if is_conversational(inputs[0]):
                    messages = [
                        {"messages": p + c} for p, c in zip(prompts, completions)
                    ]
                    texts = [
                        apply_chat_template(x, reward_processing_class)["text"]
                        for x in messages
                    ]
                else:
                    texts = [p + c for p, c in zip(prompts, completions)]
                reward_inputs = reward_processing_class(
                    texts,
                    return_tensors="pt",
                    padding=True,
                    padding_side="right",
                    add_special_tokens=False,
                )
                reward_inputs = super()._prepare_inputs(reward_inputs)
                with torch.inference_mode():
                    rewards_per_func[:, i] = reward_func(**reward_inputs).logits[
                        :, 0
                    ]  # Shape (B*G,)
            else:
                reward_kwargs = {
                    key: []
                    for key in inputs[0].keys()
                    if key not in ["prompt", "completion"]
                }
                for key in reward_kwargs:
                    for example in inputs:
                        reward_kwargs[key].extend([example[key]] * self.num_generations)
                output_reward_func = reward_func(
                    prompts=prompts, completions=completions, **reward_kwargs
                )
                rewards_per_func[:, i] = torch.tensor(
                    output_reward_func, dtype=torch.float32, device=device
                )

        # Sum the rewards from all reward functions
        rewards = rewards_per_func.sum(dim=1)
        # Compute grouped-wise rewards
        mean_grouped_rewards = rewards.view(-1, self.num_generations).mean(dim=1)
        std_grouped_rewards = rewards.view(-1, self.num_generations).std(dim=1)
        # Normalize the rewards to compute the advantages
        mean_grouped_rewards = mean_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        std_grouped_rewards = std_grouped_rewards.repeat_interleave(
            self.num_generations, dim=0
        )
        advantages = (rewards - mean_grouped_rewards) / (std_grouped_rewards + 1e-4)
        if self.use_grpo:
            # x - x.detach() allows for preserving gradients from x
            per_token_loss = torch.exp(
                per_token_logps - per_token_logps.detach()
            ) * advantages.unsqueeze(1)

            if self.beta != 0.0:
                per_token_loss = -(per_token_loss - self.beta * per_token_kl)
            else:
                per_token_loss = -(per_token_loss)

            loss = (
                (per_token_loss * completion_mask).sum(dim=1)
                / completion_mask.sum(dim=1)
            ).mean()

        else:
            coef_1 = torch.exp(per_token_logps - per_token_logps.detach())
            coef_2 = torch.clamp(coef_1, 1 - self.epsilon_low, 1 + self.epsilon_high)
            per_token_loss1 = coef_1 * advantages.unsqueeze(1)
            per_token_loss2 = coef_2 * advantages.unsqueeze(1)
            per_token_loss = -torch.min(per_token_loss1, per_token_loss2)
            if self.beta != 0.0:
                per_token_loss = per_token_loss + self.beta * per_token_kl
            loss = (per_token_loss * completion_mask).sum() / completion_mask.sum()

        # Log the metrics
        completion_length = (
            self.accelerator.gather_for_metrics(completion_mask.sum(1))
            .float()
            .mean()
            .item()
        )
        self._metrics["completion_length"].append(completion_length)
        reward_per_func = self.accelerator.gather_for_metrics(rewards_per_func).mean(0)
        for i, reward_func in enumerate(self.reward_funcs):
            if isinstance(reward_func, PreTrainedModel):
                reward_func_name = reward_func.config._name_or_path.split("/")[-1]
            else:
                reward_func_name = reward_func.__name__
            self._metrics[f"rewards/{reward_func_name}"].append(
                reward_per_func[i].item()
            )
        self._metrics["reward"].append(
            self.accelerator.gather_for_metrics(rewards).mean().item()
        )
        self._metrics["reward_std"].append(
            self.accelerator.gather_for_metrics(std_grouped_rewards).mean().item()
        )
        if self.beta != 0.0:
            mean_kl = (
                (per_token_kl * completion_mask).sum(dim=1) / completion_mask.sum(dim=1)
            ).mean()
            self._metrics["kl"].append(
                self.accelerator.gather_for_metrics(mean_kl).mean().item()
            )
        completion_lengths = completion_mask.sum(dim=1).clamp(min=1)
        mean_completion_entropy = (entropy_completion * completion_mask).sum(
            dim=1
        ) / completion_lengths
        # calc mean entropy per batch
        batch_mean_entropy = mean_completion_entropy.mean()
        self._metrics["generation_entropy"].append(
            self.accelerator.gather_for_metrics(batch_mean_entropy).mean().item()
        )

        if torch.cuda.is_available():
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
