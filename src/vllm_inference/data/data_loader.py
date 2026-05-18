import json
import os
import glob

import datasets
import pandas as pd
import yaml


def load_dataset_config(config_path="dataset/config.yaml") -> dict:
    with open(config_path, "r") as f:
        return yaml.safe_load(f)


def load_activitynet(split="test", cfg=None):
    data_path = cfg["test_path"] if cfg else "dataset/activitynet/activitynet_test.json"
    video_folder = cfg["video_folder"] if cfg else "video_datasets/ActivityNet"
    data = json.load(open(data_path))

    qid, conv_data = 0, []
    for video_id, meta_data in data.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        if not os.path.exists(video_path):
            print(f"Skip the {video_path}")
            continue

        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"anet_{qid}",
                }
            )
            qid += 1

    return conv_data

def load_charades(split="test", cfg=None):
    anno_path = cfg["test_path"] if cfg else "dataset/charades/charades_test.json"
    video_folder = cfg["video_folder"] if cfg else "video_datasets/charades"
    data = json.load(open(anno_path))

    qid, conv_data = 0, []
    for video_id, meta_data in data.items():
        video_path = os.path.join(video_folder, f"{video_id}.mp4")
        for i in range(len(meta_data["timestamps"])):
            conv_data.append(
                {
                    "video": video_path,
                    "duration": meta_data["duration"],
                    "timestamp": meta_data["timestamps"][i],
                    "sentence": meta_data["sentences"][i].strip(),
                    "qid": f"charades_{qid}",
                }
            )
            qid += 1

    return conv_data

def load_tvgbench(split="default", cfg=None):
    """
    Load JSON data in TVGBench format.

    Args:
        split (str): Dataset split (unused for TVGBench, kept for API consistency).
        cfg (dict): Dataset config entry from config.yaml (keys: test_path, video_folder).

    Returns:
        list: A list of dicts with keys: video, duration, timestamp, sentence, qid, start, end.
    """
    data_path = cfg["test_path"] if cfg else "dataset/tvgbench/tvgbench.json"

    with open(data_path, "r") as f:
        raw_data = json.load(f)

    qid_counter = 0
    conv_data = []

    for item in raw_data:
        video_path = item["path"]

        if not os.path.exists(video_path):
            print(f"{video_path} does not exist")
            continue

        duration = item["duration"]
        answer_str = item["answer"]
        question_str = item["question"]
        start = item["start"]
        end = item["end"]

        parts = answer_str.split("-")
        start_time = float(parts[0])
        end_time = float(parts[1])
        timestamp = [start_time, end_time]

        sentence = question_str

        source_prefix = "unknown"
        if "source" in item and isinstance(item["source"], str):
            source_filename = os.path.basename(item["source"])
            source_prefix = (
                os.path.splitext(source_filename)[0].replace(".", "_").replace("-", "_")
            )

        qid_str = f"{source_prefix}_{qid_counter}"
        qid_counter += 1

        conv_data.append(
            {
                "video": video_path,
                "duration": duration,
                "timestamp": timestamp,
                "sentence": sentence,
                "qid": qid_str,
                "start": start,
                "end": end,
            }
        )

    if len(conv_data) == 0:
        raise FutureWarning("The scale of evaluation data is 0.")

    return conv_data

def load_videomme(split="default", cfg=None):
    if split in ["test", "train"]:
        split = "default"
    assert split in ["short", "medium", "long", "default"]
    data_path = cfg["test_path"] if cfg else "video_datasets/Video-MME/videomme"
    video_folder = cfg["video_folder"] if cfg else "video_datasets/Video-MME/data"
    conv_data = []
    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )
    for itm in data:
        if split == "default" or itm["duration"] == split:
            video_path = os.path.join(video_folder, itm["videoID"] + ".mp4")
            if not os.path.exists(video_path):
                print(f"{video_path} does not exist")
                continue
            conv_data.append(
                {
                    "video": video_path,
                    "question": itm["question"],
                    "options": [op[2:].strip() for op in itm["options"]],
                    "answer": ord(itm["answer"]) - ord("A"),
                    "duration": None,
                    "qid": f'videomme_{itm["question_id"]}',
                }
            )

    return conv_data

def load_rextimeTG(split="default", cfg=None):
    anno_path = cfg["test_path"] if cfg else "video_datasets/ReXTime/data"
    video_folder = cfg["video_folder"] if cfg else "video_datasets"
    conv_data = []
    data = datasets.load_dataset(anno_path, split="test")

    missing_cnt = 0
    for itm in data:
        if itm["source"] == "qvhighlights_val":
            video_path = os.path.join(video_folder, "qv", itm["vid"] + ".mp4")
        else:
            video_path = os.path.join(video_folder, "ActivityNet", itm["vid"] + ".mp4")
            if not os.path.exists(video_path):
                video_path = os.path.join(video_folder, "ActivityNet", itm["vid"] + ".mkv")

        if not os.path.exists(video_path):
            print("Pass video path: ", video_path)
            missing_cnt += 1
            continue

        conv_data.append(
            {
                "video": video_path,
                "sentence": itm["question"],
                "timestamp": itm["span"],
                "duration": None,
                "qid": itm["qid"],
            }
        )

    print(f"{missing_cnt} data missing!")
    return conv_data

def load_tempcompass(split="default", cfg=None):
    if split in ["test", "train", "default"]:
        split = "multi-choice"
    assert split in ["multi-choice", "captioning", "caption_matching", "yes_no"]
    data_root = cfg["test_path"] if cfg else "video_datasets/TempCompass"
    video_folder = cfg["video_folder"] if cfg else "video_datasets/TempCompass/videos"
    data_path = os.path.join(data_root, split)

    data = datasets.load_dataset(
        "parquet", split="test", data_dir=data_path, streaming=True
    )

    conv_data = []
    for idx, itm in enumerate(data):
        key = itm["video_id"]
        dim = itm["dim"]
        video_path = os.path.join(video_folder, key + ".mp4")
        question, options, answer = itm["question"], [], itm["answer"]

        if split == "yes_no":
            options = ["yes", "no"]
            answer = options.index(answer)
        elif split == "caption_matching":
            tmp = question.split("\n")
            question = tmp[0]
            options = [":".join(line.split(":")[1:]).strip() for line in tmp[1:]]
            answer = options.index(":".join(answer.split(":")[1:]).strip())
        elif split == "multi-choice":
            tmp = question.split("\n")
            question = tmp[0]
            options = [line[2:].strip() for line in tmp[1:]]
            answer = ord(answer[0]) - ord("A")

        conv_data.append(
            {
                "video": video_path,
                "question": question,
                "options": options,
                "answer": answer,
                "duration": None,
                "qid": f"tempcompass|{split}|{key}|{dim}|{idx}",
            }
        )

    return conv_data

def load_timer1(dataset_path, split="default"):
    """
    Load JSON data in TVGBench format.

    Args:
        dataset_path (str): Path to the JSON file in TimeR1 format.

    Returns:
        list: A list containing processed data, where each element is a dictionary
            in the format {'video': str, 'duration': float, 'timestamp': list[float, float], 'sentence': str, 'qid': str}.
            Returns an empty list if the file does not exist or cannot be parsed.
    """

    # captioning
    if "window" in dataset_path:
        with open(dataset_path, "r") as f:
            raw_data = json.load(f)

        conv_data = []

        for item in raw_data:
            video_path = item["video"]
            duration = item["duration"]
            start = item.get("video_start", 0)
            end = item.get("video_end", duration)
            timestamps = item["timestamps"]
            qid = item["qid"]

            for timestamp in timestamps:
                conv_data.append(
                    {
                        "video": video_path,
                        "duration": duration,
                        "qid": qid,
                        "video_start": start,
                        "video_end": end,
                        "timestamp": timestamp,
                    }
                )

    # sampling windows
    else:
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
                }
            )

    if len(conv_data) == 0:
        raise FutureWarning("The scale of evaluation data is 0.")

    return conv_data

def load_etbenchTG(split="default", cfg=None):
    data_path = cfg["test_path"] if cfg else "dataset/ETBench/etbench_vid_v1.0.json"
    video_folder = cfg["video_folder"] if cfg else "video_datasets/ETBench/videos"
    conv_data = []
    data = json.load(open(data_path))
    data = [item for item in data if item["task"] == "tvg"]

    for item in data:
        video_path = os.path.join(video_folder, f"{item['video']}")

        if not os.path.exists(video_path):
            print(f"Skip the {video_path}")
            continue

        conv_data.append(
            {
                "video": video_path,
                "duration": item['duration'],
                "timestamp": item["tgt"],
                "sentence": item['q'].split("'")[1],
                "qid": f"etbench|tvg|{item['idx']}",
            }
        )

    if len(conv_data) == 0:
        raise FutureWarning("The scale of evaluation data is 0.")

    return conv_data

def _extract_qid(itm):
    vtype, vid, question = (
        None,
        itm["video"].split("/")[-1].split(".")[0],
        itm["sentence"],
    )
    video_path = itm["video"].lower()
    if "cosmo" in video_path or "howto100m" in video_path:
        vtype = "cosmo"
    if "queryd" in video_path:
        vtype = "queryd"
    if "vtime" in video_path:
        vtype = "internvid-vtime"
        if ":" in vid:
            vid = vid.split(":")[0][:-3]
    if "didemo" in video_path:
        vtype = "didemo"
    if "yt_temporal_videos" in video_path:
        vtype = "yt-temporal"

    return f"my|{vtype}|{vid}|{question}"
