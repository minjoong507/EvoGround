import argparse
import json
import os
import re
import time
import numpy as np
import ast
import glob
from numbers import Number
import yaml

import torch
import torch.nn.functional as F
from tqdm import tqdm
from transformers import AutoProcessor
from src.model.model_utils import get_model_type
from src.vllm_inference.gen_data import build_dataloader
from src.vllm_inference.vllm_infer import vllmWrapper
from src.utils import extract_windows_for_gen_data

from src.prompts import SAMPLING_WINDOW_TEMPLATE_v2, JUDGE_PROMPT, DENSE_VIDEO_CAPTIONING_TEMPLATE, \
    SAMPLING_WINDOW_TEMPLATE_v3

PROMPT_DICT = {
    "v2": SAMPLING_WINDOW_TEMPLATE_v2,
    "v3": SAMPLING_WINDOW_TEMPLATE_v3,
    "judge": JUDGE_PROMPT
}

def resolve_latest_checkpoint(path: str) -> str:
    # If user already passed a checkpoint-* dir, keep it
    if os.path.isdir(path) and re.search(r"checkpoint-\d+$", os.path.basename(path)):
        return path

    ckpts = glob.glob(os.path.join(path, "checkpoint-*"))
    if not ckpts:
        return path  # no checkpoints found; keep as-is

    def step_of(p):
        m = re.search(r"checkpoint-(\d+)$", os.path.basename(p))
        return int(m.group(1)) if m else -1

    return max(ckpts, key=step_of)

def save_json(data_list, output_path):
    """Helper function to save a list to a JSON file."""
    if data_list and isinstance(data_list[0], dict) and "data" in data_list[0]:
        data_to_save = [item["data"] for item in data_list]
    else:
        data_to_save = data_list
    if not data_to_save:
        return

    output_dir = os.path.dirname(output_path)
    if output_dir and not os.path.exists(output_dir):
        os.makedirs(output_dir)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(data_to_save, f, indent=4, ensure_ascii=False)
        print(f"save to: {output_path}")


def get_args():
    parser = argparse.ArgumentParser(
        description="Evaluation for training-free video temporal grounding (Single GPU Version)"
    )
    parser.add_argument(
        "--model_base", type=str, default="../pretrained_models/Qwen2.5-VL-7B-Instruct"
    )
    parser.add_argument("--batch_size", type=int, default=1, help="Batch size")
    parser.add_argument("--max_windows", type=int, default=4, help="")
    parser.add_argument(
        "--output_dir",
        type=str,
        default="checkpoints",
        help="Directory to save checkpoints",
    )
    parser.add_argument(
        "--device", type=str, default="cuda:0", help="GPU device to use"
    )
    parser.add_argument(
        "--pipeline_parallel_size", type=int, default=1, help="GPU nodes"
    )
    parser.add_argument("--split", type=str, default="train", help="dataset type")
    parser.add_argument("--max_new_tokens", type=int, default=1024)
    parser.add_argument("--curr_idx", type=int, default=0, help="数据分片")
    parser.add_argument("--total_idx", type=int, default=1, help="数据分片")
    parser.add_argument(
        "--total_pixels", type=int, default=3584 * 28 * 28, help="total_pixels"
    )
    parser.add_argument(
        "--datasets",
        nargs="+",
        type=str,
        help="dataset names",
        default=["timer1"],
    )
    parser.add_argument(
        "--use_vllm_inference", action="store_true"
    )
    parser.add_argument(
        "--use_think", action="store_true"
    )

    parser.add_argument("--prompt_type", type=str, default="v2", help="Prompt type", choices=list(PROMPT_DICT.keys()))
    parser.add_argument(
        "--use_prepared_video",
        action="store_true",
        help="Use video cache in ./video_cache",
    )
    parser.add_argument(
        "--dataset_info",
        type=str,
        default="./dataset/config.yaml",
    )
    # --- distribution-shift sampling ---
    parser.add_argument(
        "--num_generations",
        type=int,
        default=1,
        help="Number of generation paths per video. >1 enables diversity-based selection.",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="Sampling temperature. Must be >0 when num_generations>1 to get diverse outputs.",
    )
    parser.add_argument(
        "--prev_gen_data_path",
        type=str,
        default=None,
        help="Path to a previously generated gen_data JSON. "
             "When provided and num_generations>1, selects the most novel candidate "
             "relative to the previous generation's window distribution.",
    )

    return parser.parse_args()

def build_model(args):
    model_type = getattr(args, "model_type", None) or get_model_type(args.model_base)
    args.model_type = model_type
    processor = AutoProcessor.from_pretrained(args.model_base, use_fast=True)
    processor.tokenizer.padding_side = "left"
    model = vllmWrapper(args)
    return model, processor


# ---------------------------------------------------------------------------
# Distribution-shift sampling helpers
# ---------------------------------------------------------------------------

def load_prev_gen_data(path: str) -> dict[str, list[list[float]]]:
    """Load previous gen_data JSON and index windows by video path.

    Returns:
        dict mapping video_path -> list of [start, end] windows
    """
    if path is None or not os.path.exists(path):
        return {}

    with open(path, "r") as f:
        records = json.load(f)

    index: dict[str, list] = {}
    for rec in records:
        video = rec.get("video", "")
        ts = rec.get("timestamp")
        if video and ts and len(ts) == 2:
            index.setdefault(video, []).append(ts)
    return index


def _window_center(window: list[float], duration: float) -> float:
    """Normalized center of a temporal window in [0, 1]."""
    return ((window[0] + window[1]) / 2.0) / max(duration, 1e-6)


def novelty_score(
    candidate_windows: list[list[float]],
    prev_windows: list[list[float]],
    duration: float,
) -> float:
    """Measure how novel a candidate generation is relative to previous windows.

    For each candidate window, compute the minimum L1 center-distance to any
    previous window (both normalized to [0, 1] by duration).  Returns the mean
    over all candidate windows — higher means more novel / distribution-shifted.

    If there are no previous windows the candidate is trivially fully novel (1.0).
    """
    if not prev_windows or not candidate_windows:
        return 1.0

    prev_centers = [_window_center(w, duration) for w in prev_windows]
    total = 0.0
    for w in candidate_windows:
        c = _window_center(w, duration)
        total += min(abs(c - pc) for pc in prev_centers)
    return total / len(candidate_windows)


def select_best_candidate(
    candidates: list[tuple[list, list]],
    prev_windows: list[list[float]],
    duration: float,
) -> int:
    """Return the index of the candidate with the highest novelty score."""
    best_idx, best_score = 0, -1.0
    for i, (windows, _) in enumerate(candidates):
        score = novelty_score(windows, prev_windows, duration)
        if score > best_score:
            best_score = score
            best_idx = i
    return best_idx

@torch.no_grad()
def inference(model, inputs):
    for key in inputs.keys():
        if not isinstance(inputs[key], torch.Tensor):
            continue
        inputs[key] = inputs[key].to(model.device)

    logits = model(**inputs).logits
    bsz, seq_len, _ = logits.shape
    if "attention_mask" in inputs:
        pred_token_indices = torch.sum(inputs["attention_mask"], dim=-1) - 1
    else:
        pred_token_indices = torch.full((bsz,), seq_len - 1, device=logits.device)

    pred_token_logits = logits[
        torch.arange(bsz, device=logits.device), pred_token_indices, :
    ]

    return pred_token_logits


@torch.no_grad()
def calc_prob(logits, options_token_ids):
    bsz = logits.shape[0]
    probs = []
    for i in range(bsz):
        logit = logits[i, options_token_ids]
        probs.append(F.softmax(logit, dim=1))
    return probs


@torch.no_grad()
def main(args):
    os.makedirs(args.output_dir, exist_ok=True)
    output_file = os.path.join(
        args.output_dir, f"gen_data_{args.curr_idx}_{args.total_idx}.json"
    )

    if args.num_generations > 1 and args.temperature <= 0.0:
        print(
            "[WARNING] num_generations > 1 but temperature <= 0.0. "
            "All generations will be identical. Set --temperature > 0 for diverse outputs."
        )

    with open(args.dataset_info, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)

    dataset_name = args.datasets[0]

    if dataset_name not in dataset_config:
        raise NotImplementedError(f"Dataset {dataset_name} is not supported.")

    args.dataset_path = dataset_config[dataset_name]["anno_path"]
    args.video_folder = dataset_config[dataset_name]["video_folder"]

    already_finished = set([])

    model, processor = build_model(args)
    prompt = PROMPT_DICT[args.prompt_type]

    dataloader_args = {
        "prompt": prompt,
        "max_windows": args.max_windows,
        "batch_size": args.batch_size,
        "already_finished": already_finished,
        "curr_idx": args.curr_idx,
        "total_idx": args.total_idx,
        "split": args.split,
        "num_workers": min(8, args.batch_size),
        "dataset_names": args.datasets,
        "use_prepared_video": args.use_prepared_video,
        "total_pixels": args.total_pixels,
        "dataset_path": args.dataset_path,
        "video_folder": args.video_folder,
    }

    dataloader = build_dataloader(processor, **dataloader_args)

    program_start_time = time.perf_counter()

    data_list = []
    skip_cnt = 0

    for batch_itm in tqdm(dataloader):
        # output_texts: List[List[str]]  shape [batch_size, num_generations]
        output_texts = model.generate(
            batch_itm["inputs"],
            max_new_tokens=args.max_new_tokens,
            num_generations=args.num_generations,
            temperature=args.temperature,
        )

        for i in range(len(output_texts)):
            video_path = batch_itm["video_paths"][i]
            duration = batch_itm["duration"][i] if "duration" in batch_itm else None

            if dataset_name in ["temporalbench"]:
                # For temporalbench we only use the first generation (no timestamp selection needed)
                windows, captions = extract_windows_for_gen_data(output_texts[i][0], duration=duration)
                if captions is None:
                    print("Failed to extract a caption.")
                    skip_cnt += 1
                    continue

                video_caption = " ".join(captions)
                data_list.append({
                    "video": video_path,
                    "qid": batch_itm["qid"][i],
                    "response": video_caption,
                    "duration": duration,
                })

            elif dataset_name in ["timer1", "etbench", "charades", "activitynet", "activitynet_dvc"]:
                # Parse all candidate generations
                candidates = []
                for gen_text in output_texts[i]:
                    windows, captions = extract_windows_for_gen_data(gen_text, duration=duration)
                    if windows is not None and captions is not None:
                        candidates.append((windows, captions))

                if not candidates:
                    print("Failed to extract windows and captions from any generation.")
                    skip_cnt += 1
                    continue

                # # Select the most novel candidate relative to the previous generation
                # if len(candidates) > 1 and prev_data_by_video:
                #     prev_windows = prev_data_by_video.get(video_path, [])
                #     best_idx = select_best_candidate(candidates, prev_windows, duration or 1.0)
                # else:
                #     best_idx = 0
                # Hack
                # windows, captions = candidates[best_idx]

                windows, captions = candidates[0]
                records = [
                    {
                        "video": video_path,
                        "qid": batch_itm["qid"][i],
                        "timestamp": window,
                        "sentence": caption,
                        "duration": duration,
                    }
                    for window, caption in zip(windows, captions)
                ]
                data_list.extend(records)

            else:
                raise NotImplementedError(f"Check the dataset name. {dataset_name} is not supported.")

    # --- END TOTAL TIME & CALCULATIONS ---
    program_end_time = time.perf_counter()
    total_program_duration = program_end_time - program_start_time

    print("\n--- Timing Summary ---")
    print(f"Total program execution time: {total_program_duration:.2f} seconds")

    print(f"{skip_cnt} videos are skipped. {len(data_list)} data samples will be saved in {output_file}")
    save_json(data_list, output_file)


if __name__ == "__main__":
    from src.vllm_inference.utils import monkey_patch

    monkey_patch()
    args = get_args()
    main(args)
