import json
import os
import glob
import re

import datasets
import pandas as pd

def load_activitynet_dvc(dataset_path, video_folder):
    """Load one record per GT caption — used for proposer training data generation."""
    data = json.load(open(dataset_path))

    qid, conv_data = 0, []
    for video_id, meta_data in data.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, f"{video_id}.mkv")

        if not os.path.exists(video_path):
            continue

        duration = meta_data["duration"]
        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": duration,
                    "video_start": 0,
                    "video_end": duration,
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"anet_{qid}",
                }
            )
            qid += 1

    return conv_data


def load_activitynet_dvc_eval(dataset_path, video_folder):
    """Load one record per unique video — used for DVC evaluation."""
    data = json.load(open(dataset_path))

    conv_data = []
    for video_id, meta_data in data.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            video_path = os.path.join(video_folder, f"{video_id}.mkv")

        if not os.path.exists(video_path):
            continue

        duration = meta_data["duration"]
        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "video_start": 0,
                "video_end": duration,
                "qid": video_id,  # use video_id directly for easy submission grouping
            }
        )

    return conv_data

def load_timer1(dataset_path, video_folder, w_timestamp=False):
    """
    Load JSON data in TVGBench format.

    Args:
        data_path (str): Path to the JSON file in TimeR1 format.

    Returns:
        list: A list containing processed data, where each element is a dictionary
            in the format {'video': str, 'duration': float, 'timestamp': list[float, float], 'sentence': str, 'qid': str}.
            Returns an empty list if the file does not exist or cannot be parsed.
    """

    # sampling windows
    with open(dataset_path, "r") as f:
        raw_data = json.load(f)

    qid_counter = 0
    conv_data = []

    for item in raw_data:
        video_path = item["video"]

        if not os.path.exists(video_path):
            print(f"{video_path} does not exist")
            continue

        duration = item["duration"]
        start = item.get("video_start", 0)
        end = item.get("video_end", duration)

        qid_str = f"timer1_{qid_counter}"
        qid_counter += 1

        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "qid": qid_str,
                "video_start": start,
                "video_end": end,
                "timestamp": item["timestamp"],
                "sentence": item["sentence"]
            } if w_timestamp else
            {
                "video": video_path,
                "duration": duration,
                "qid": qid_str,
                "video_start": start,
                "video_end": end,
            }
        )

    if len(conv_data) == 0:
        raise FutureWarning("The scale of evaluation data is 0.")

    return conv_data

def load_charades(dataset_path, video_folder):
    conv_data = []
    data = json.load(open(dataset_path))
    # data = json.load(open(dataset_path))

    cur_idx = 0

    for item in data:
        video_path = os.path.join(video_folder, f"{item['video']}")

        if not os.path.exists(video_path):
            print(f"Skip the {video_path}")
            continue

        duration = item["duration"]
        start = item.get("video_start", 0)
        end = item.get("video_end", duration)
        idx = item.get("idx", -1)

        if idx == -1:
            idx = cur_idx
            cur_idx += 1

        conv_data.append(
            {
                "video": video_path,
                "duration": item['duration'],
                "sentence": item['sentence'],
                "qid": f"charades|{idx}",
                "video_start": start,
                "video_end": end,
            }
        )

    return conv_data


def load_etbenchDVC(dataset_path, video_folder, split="default"):
    conv_data = []
    # data = json.load(open("./dataset/ETBench/annotations/etbench_vid_v1.0.json"))
    data = json.load(open(dataset_path))
    data = [item for item in data if item["task"] in ["dvc", "slc"]]

    for item in data:
        # video_path = os.path.join(f"video_datasets/ETBench/videos", f"{item['video']}")
        video_path = os.path.join(video_folder, f"{item['video']}")

        if not os.path.exists(video_path):
            print(f"Skip the {video_path}")
            continue

        duration = item["duration"]
        start = item.get("video_start", 0)
        end = item.get("video_end", duration)

        conv_data.append(
            {
                "video": video_path,
                "duration": item['duration'],
                "sentence": item['q'].split('For each step,')[0],
                "qid": f"etbench|{item['task']}|{item['idx']}",
                "video_start": start,
                "video_end": end,
            }
        )

    return conv_data

def load_temporalbench(dataset_path, video_folder):
    def parse_start_end_seconds(path):
        pattern = r"_start_(\d+(?:\.\d+)?)_end_(\d+(?:\.\d+)?)"
        match = re.search(pattern, path)

        if not match:
            raise ValueError(f"Could not parse start/end from: {path}")

        start = float(match.group(1))
        end = float(match.group(2))

        return start, end

    conv_data = []
    data = json.load(open(dataset_path))

    for item in data:
        # video_path = os.path.join(f"video_datasets/TemporalBench/", f"{item['video_name']}")
        video_path = os.path.join(video_folder, f"{item['video_name']}")

        if not os.path.exists(video_path):
            print(f"Skip the {video_path}")
            continue

        duration = item["duration"]

        if duration < 3:
            print(f"Skip the video: {video_path} due to very short duration.")
            continue

        start = item.get("video_start", 0)
        end = item.get("video_end", duration)

        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "qid": f"temporalbench|{item['idx']}",
                "video_start": start,
                "video_end": end,
            }
        )

    return conv_data
