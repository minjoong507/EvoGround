import argparse
import json
import os
import re
import time
import numpy as np
import ast
import glob
import yaml
from numbers import Number


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
        "--gen_dataset_path",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
    )

    parser.add_argument(
        "--output_file_name",
        type=str,
        default="gen_data.json",
    )

    parser.add_argument(
        "--dataset_info",
        type=str,
        default="dataset/config.yaml",  # Replace with your actual video folder path
    )

    return parser.parse_args()

def main(args):
    with open(args.dataset_info, "r", encoding="utf-8") as f:
        dataset_config = yaml.safe_load(f)

    gen_data_paths = glob.glob(os.path.join(args.gen_dataset_path, "gen_data_*.json"))
    # gen_data_paths = glob.glob(os.path.join(args.gen_dataset_path, "gen_data_w_score_*.json"))

    if len(gen_data_paths) < 1:
        raise FileNotFoundError(f"Check the path {args.gen_dataset_path} has right json files.")
    else:
        print("Will load gen data from ", gen_data_paths)

    if args.dataset in ["timer1"]:
        org_dataset_path = dataset_config[args.dataset]["anno_path"]
        with open(org_dataset_path, "r", encoding="utf-8") as f:
            org_data = json.load(f)

        gen_data = []

        for gen_data_path in gen_data_paths:
            with open(gen_data_path, "r", encoding="utf-8") as f:
                gen_data += json.load(f)

        video_id_to_start_end = {}

        for _org_data in org_data:
            video_id_to_start_end[_org_data["video"]] = {
                "video_start": _org_data["video_start"],
                "video_end": _org_data["video_end"]
            }

        for _gen_data in gen_data:
            if _gen_data["video"] in video_id_to_start_end and "video_start" not in _gen_data and "video_end" not in _gen_data:
                _gen_data["video_start"] = video_id_to_start_end[_gen_data["video"]]["video_start"]
                _gen_data["video_end"] = video_id_to_start_end[_gen_data["video"]]["video_end"]

    else:
        gen_data = []

        for gen_data_path in gen_data_paths:
            with open(gen_data_path, "r", encoding="utf-8") as f:
                gen_data += json.load(f)


    # save_json(gen_data, os.path.join(args.gen_dataset_path, "gen_data_w_score.json"))
    print("total number of data: ", len(gen_data))
    save_json(gen_data, os.path.join(args.gen_dataset_path, args.output_file_name))

if __name__ == "__main__":
    args = get_args()
    main(args)
