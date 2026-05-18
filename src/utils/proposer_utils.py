import os
import random
import ast
from numbers import Number
import re
import sys

DUMMY_WINDOW = [0, 0]
PADDING_WINDOW = [0, int(sys.maxsize)]


def parse_partial_output(time_text, caption_text):
    """
    Robustly parse time/caption blocks that may be truncated.
    First tries ast.literal_eval; on failure falls back to regex extraction
    of individual [start, end] pairs and quoted strings, then zips them.
    Returns (list_of_windows, list_of_captions) — lengths may differ from
    the original if truncation cut off some entries.
    """
    def _parse_times(text):
        try:
            result = ast.literal_eval(text.strip())
            if isinstance(result, list):
                return result
        except Exception:
            pass
        # Fallback: extract all [number, number] pairs
        return [
            [float(a), float(b)]
            for a, b in re.findall(r"\[\s*([\d.]+)\s*,\s*([\d.]+)\s*\]", text)
        ]

    def _parse_captions(text):
        try:
            result = ast.literal_eval(text.strip())
            if isinstance(result, list):
                return [str(c) for c in result]
        except Exception:
            pass
        # Fallback: extract all single- or double-quoted strings
        return re.findall(r'"([^"]*?)"|\'([^\']*?)\'', text)

    times = _parse_times(time_text)
    raw_caps = _parse_captions(caption_text)
    # re.findall with two groups returns tuples; flatten to strings
    if raw_caps and isinstance(raw_caps[0], tuple):
        captions = [a or b for a, b in raw_caps]
    else:
        captions = raw_caps

    # Zip to the shorter length so indices always align
    return times, captions

def extract_windows(text, inputs, max_windows=5, min_length=5):
    match = re.search(r"<answer>\s*(\[.*?\])\s*</answer>", text, re.S)
    random_windows = [sorted(random.sample(range(int(inputs[0]["durations"]) - 3), 2)) for _ in range(max_windows)]
    random_windows = [[target_window[0], target_window[1] + 2] for target_window in random_windows]

    if not match:
        print("Window proposal failed.")
        return random_windows

    else:
        try:
            print("Window proposal succeeded.")
            windows = ast.literal_eval(match.group(1))
            windows = [window for window in windows if len(window) == 2]
            windows = [window for window in windows if
                       isinstance(window[0], Number) and isinstance(window[1], Number) and min_length <= window[1] - window[
                           0]]
            windows = [window for window in windows if
                       0 <= window[0] < inputs[0]["durations"] and 0 <= window[1] < inputs[0]["durations"]]

            # Padding
            windows += random_windows
            windows = windows[:max_windows]

        except:
            print("Randomly sample windows")
            return random_windows

    return windows

def check_if_window_availabie(win, duration, min_length=3):
    if not (isinstance(win, (list, tuple)) and len(win) == 2):
        return 0.0, "invalid_window_format"

    if not (isinstance(win[0], Number) and isinstance(win[1], Number)):
        return 0.0, "invalid_time_type"

    if not (min_length <= (win[1] - win[0])):
        return 0.0, "too_short_window"

    if not (0 <= win[0] < win[1] <= duration):
        return 0.0, "over_duration"

    return 1.0, "success"


def extract_windows2(text, duration, max_window, total_window=8, min_length=3, penalize=False):
    if os.getenv("DEBUG_MODE") == "true":
        print("model output: ", text)

    time_match = re.search(r"<time>(.*?)</time>", text, re.DOTALL)
    caption_match = re.search(r"<description>(.*?)</description>", text, re.DOTALL)

    extracted_pairs = []
    valid_cnt = 0

    # Also try to match truncated output (no closing tag)
    if not caption_match:
        caption_match = re.search(r"<description>(.*)", text, re.DOTALL)
    if not time_match:
        time_match = re.search(r"<time>(.*)", text, re.DOTALL)

    if time_match and caption_match:
        parsed_times, parsed_captions = parse_partial_output(
            time_match.group(1), caption_match.group(1)
        )

        if isinstance(parsed_times, list) and isinstance(parsed_captions, list) and len(parsed_times) > 0:
            for win, cap in zip(parsed_times, parsed_captions):
                if (
                    isinstance(win, (list, tuple))
                    and len(win) == 2
                    and isinstance(win[0], Number)
                    and isinstance(win[1], Number)
                    and min_length <= (win[1] - win[0])
                    and 0 <= win[0] < win[1] <= duration
                ):
                    extracted_pairs.append((win, cap))
                    valid_cnt += 1

                else:
                    extracted_pairs.append((DUMMY_WINDOW, cap))

    # Penalize for producing fewer than max_window valid windows
    while len(extracted_pairs) < max_window:
        extracted_pairs.append((DUMMY_WINDOW, "DUMMY"))

    random.shuffle(extracted_pairs)
    selected_pairs = extracted_pairs[:total_window]

    # Fill remaining slots up to total_window with PADDING (no penalization)
    while len(selected_pairs) < total_window:
        selected_pairs.append((PADDING_WINDOW, "PADDING"))

    valid_windows, valid_captions = zip(*selected_pairs)
    print(f"{valid_cnt} valid pairs out of {len(selected_pairs)} total.")

    return list(valid_windows), list(valid_captions)


def extract_windows_for_gen_data(text, duration):
    if os.getenv("DEBUG_MODE") == "true":
        print("model output: ", text)

    time_match = re.search(r"<time>(.*?)</time>", text, re.DOTALL)
    caption_match = re.search(r"<description>(.*?)</description>", text, re.DOTALL)

    valid_pairs = []
    success = False

    # Also try to match truncated output (no closing tag)
    if not caption_match:
        caption_match = re.search(r"<description>(.*)", text, re.DOTALL)
    if not time_match:
        time_match = re.search(r"<time>(.*)", text, re.DOTALL)

    if time_match and caption_match:
        try:
            parsed_times, parsed_captions = parse_partial_output(
                time_match.group(1), caption_match.group(1)
            )
            # parsed_times = ast.literal_eval(time_match.group(1).strip())
            # parsed_captions = ast.literal_eval(caption_match.group(1).strip())

            if (
                isinstance(parsed_times, list)
                and isinstance(parsed_captions, list)
                and len(parsed_times) == len(parsed_captions)
                and len(parsed_times) > 0
            ):
                for win, cap in zip(parsed_times, parsed_captions):
                    if (
                        isinstance(win, (list, tuple))
                        and len(win) == 2
                        and isinstance(win[0], Number)
                        and isinstance(win[1], Number)
                        and 0 <= win[0] < win[1] <= duration
                    ):
                        valid_pairs.append((win, cap))

                if len(valid_pairs) > 0:
                    success = True

        except Exception as e:
            print(f"Parsing failed: {e}")

    if not success:
        return None, None

    valid_windows, valid_captions = zip(*valid_pairs)

    return list(valid_windows), list(valid_captions)
