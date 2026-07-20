import re
from datetime import datetime
from typing import List, Optional

def parse_timestamp_output(output_string):
    """Parses timestamp output, similar to the example code."""
    # 1. Find all <answer>...</answer> blocks.
    answer_matches = re.findall(r"<answer>(.*?)</answer>", output_string, re.DOTALL)

    if not answer_matches:
        return None  # No <answer> tags found.

    # 2. Use the content of the *last* <answer> block.
    last_answer_content = answer_matches[-1]
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


def calculate_temporal_iou(w1: List[float], w2: List[float]) -> float:
    """Calculates Intersection over Union for two temporal windows."""
    start = max(w1[0], w2[0])
    end = min(w1[1], w2[1])
    intersection = max(0, end - start)
    union = (w1[1] - w1[0]) + (w2[1] - w2[0]) - intersection
    return intersection / (union + 1e-8)


from .reward import (
    proposer_feedback,
    proposer_format_reward,
    proposer_format_reward_gdpo,
)


__all__ = [
    "proposer_feedback",
    "proposer_format_reward",
    "proposer_format_reward_gdpo",
]