import re
import ast
import os
from dataclasses import dataclass
from datetime import datetime
from src.reward import parse_timestamp_output
from src.utils.proposer_utils import check_if_window_availabie
from src.utils import DUMMY_WINDOW, PADDING_WINDOW

def calculate_temporal_iou_from_content(content, sol, duration):
    if content is None or sol == DUMMY_WINDOW or sol == PADDING_WINDOW:
        return 0.0

    else:
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
                iou = max(intersection, 0)

            gt_start_norm = 1.0 * s / duration
            gt_end_norm = 1.0 * e / duration
            pred_start_norm = 1.0 * start_time / duration
            pred_end_norm = 1.0 * end_time / duration
            reward = (
                    iou
                    * (1 - abs(gt_start_norm - pred_start_norm))
                    * (1 - abs(gt_end_norm - pred_end_norm))
            )

        return reward


def proposer_feedback(
    solver_completions, windows_set, **kwargs
):  # Modified reward function name and arguments
    """Reward function that calculates IoU between predicted and ground truth timestamps."""
    rewards = []
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    durations = kwargs.get("durations")
    duration = durations[0]


    for completions, windows in zip(solver_completions, windows_set):
        _rewards = []
        messages = []

        for content, sol in zip(completions, windows):
            reward = 0.0

            if sol is None or sol == [0.0, 0.0]:
                _rewards.append(reward)

            elif sol == PADDING_WINDOW:
                continue # Skip.

            else:
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
                        iou = max(intersection, 0)

                    gt_start_norm = 1.0 * s / duration
                    gt_end_norm = 1.0 * e / duration
                    pred_start_norm = 1.0 * start_time / duration
                    pred_end_norm = 1.0 * end_time / duration
                    reward = (
                        iou
                        * (1 - abs(gt_start_norm - pred_start_norm))
                        * (1 - abs(gt_end_norm - pred_end_norm))
                    )

                message = f"------------- {current_time} IoU reward: {reward} -------------\n" + f"Solver Content: {content}" + f"pred second: {str(start_time)}, {str(end_time)}\n" + f"gt second: {str(gt_start)}, {str(gt_end)}\n"
                messages.append(message)

                _rewards.append(reward)


        if len(_rewards) == 0:
            rewards.append(0.0)
        else:
            rewards.append(sum(_rewards) / len(_rewards))

        print("\n".join(messages))

    return rewards

# def format_reward(completions, **kwargs):
#     """Reward function that checks if the completion has a specific format."""
#     pattern = re.compile(r"<think>.*?</think>\s*<answer>.*?</answer>", re.DOTALL)
#     matches = [re.fullmatch(pattern, content.strip()) for content in completions]
#     print("matches:", matches, completions)
#     return [1.0 if match else 0.0 for match in matches]



def proposer_format_reward(completions, **kwargs):
    """Reward function that checks if the completion has a specific format."""
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    max_windows = kwargs.get("max_windows", 10)
    durations = kwargs.get("durations")
    duration = durations[0]

    rewards = []

    for content in completions:
        # Handle conversational format: [{"role": "assistant", "content": "..."}]
        if isinstance(content, list):
            content = content[-1]["content"]
        elif isinstance(content, dict):
            content = content["content"]

        message = (
            f"------------- {current_time} ------- {duration} seconds --------- \n"
            f"Content: {content}\n"
            )

        _rewards = []
        pairs = []

        time_match = re.search(r"<time>(.*?)</time>", content, re.DOTALL)
        caption_match = re.search(r"<description>(.*?)</description>", content, re.DOTALL)

        if time_match and caption_match:
            try:
                parsed_times = ast.literal_eval(time_match.group(1).strip())
                parsed_captions = ast.literal_eval(caption_match.group(1).strip())

                if not (len(parsed_times) == len(parsed_captions) > 0):
                    message += "timestamps and captions count mismatch\n"

                else:
                    for i, (win, cap) in enumerate(zip(parsed_times, parsed_captions)):
                        if i < max_windows:
                            reward, case = check_if_window_availabie(win, duration, min_length=3)
                            pairs.append((win, cap, reward, case))
                            _rewards.append(reward)

                        else:
                            pairs.append((win, cap, 0.0, "Over the max number of windows"))
                            _rewards.append(0.0)

            except Exception as e:
                print(f"Parsing failed: {e}")
                message += "Parsing failed\n"

        final_reward = 0.0
        if len(_rewards) == len(pairs) > 0:
            final_reward = sum(_rewards) / len(_rewards)

            pairs_text = "\n".join(
                f"win={win}, caption={cap}, reward={rew}, case={cas}"
                for win, cap, rew, cas in pairs
            )

            message += (
                f"------------- Total={final_reward:.4f}\n"
                f"{pairs_text}\n"
                f"-------------\n"
            )

        rewards.append(final_reward)
        print(message)

    print("Final rewards:", rewards)

    return rewards

def proposer_format_reward_gdpo(completions, solver_completions, windows_set, **kwargs):
    current_time = datetime.now().strftime("%d-%H-%M-%S-%f")
    rewards = []
    durations = kwargs.get("durations")
    duration = durations[0]
    format_iou_thd = float(kwargs.get("format_iou_thd", 0.3))

    for content, _solver_completions, windows in zip(completions, solver_completions, windows_set):
        ious = [
            calculate_temporal_iou_from_content(solver_completion, sol, duration=duration)
            for solver_completion, sol in zip(_solver_completions, windows)
        ]

        reward_for_each = []
        valid_cnt = 0

        for i, window in enumerate(windows):
            if window == PADDING_WINDOW:
                reward_for_each.append(0)

            elif window == DUMMY_WINDOW:
                reward_for_each.append(0)
                valid_cnt += 1
            else:
                reward_for_each.append(1 if format_iou_thd <= ious[i] else 0)
                valid_cnt += 1

        reward = sum(reward_for_each) / valid_cnt if valid_cnt > 0 else 0.0


        pairs = [
            f"window: {w}, reward: {r}, iou: {iou}"
            for w, r, iou in zip(windows, reward_for_each, ious)
        ]

        message = (
            f"------------- {current_time} | Total={reward:.4f} ({len(pairs)}) | {pairs} -------------\n"
        )

        if os.getenv("DEBUG_MODE") == "true":
            print(message)

        rewards.append(reward)

    if len(rewards) != len(completions):
        return [0.0 for _ in range(len(completions))]

    return rewards