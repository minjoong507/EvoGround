import cv2
import json
import re
import os
from tqdm import tqdm


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

def parse_start_end_seconds(path):
    pattern = r"_start_(\d+(?:\.\d+)?)_end_(\d+(?:\.\d+)?)"
    match = re.search(pattern, path)

    if not match:
        raise ValueError(f"Could not parse start/end from: {path}")

    start = float(match.group(1))
    end = float(match.group(2))

    return start, end

def get_video_duration(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        raise ValueError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_count = cap.get(cv2.CAP_PROP_FRAME_COUNT)

    cap.release()

    if fps == 0:
        raise ValueError("FPS is zero, cannot compute duration")

    duration = frame_count / fps
    return duration

if __name__ == "__main__":
    if os.path.exists("dataset/temporalbench/temporalbench_short_caption_w_duration.json"):
        print("dataset/temporalbench/temporalbench_short_caption_w_duration.json file is already existed.")
    else:
        data = json.load(open('dataset/temporalbench/temporalbench_short_caption.json'))

        for idx, itm in tqdm(enumerate(data), total=len(data)):
            # try:
            #     start, end = parse_start_end_seconds(itm['video_name'])
            #     duration = float(end) - float(start)
            # except:
            video_path = os.path.join("video_datasets/TemporalBench",itm['video_name'])
            if not os.path.exists(video_path):
                raise ValueError(f"Cannot find video: {video_path}")
            duration = get_video_duration(video_path)

            itm['duration'] = duration

        save_json(data, "dataset/temporalbench/temporalbench_short_caption_w_duration.json")


