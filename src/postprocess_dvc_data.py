import argparse
import json
import os
import re
import time
import numpy as np
import ast
import glob
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
        "--org_dataset_path",
        type=str,
        default="./dataset/ETBench/annotations/etbench_vid_v1.0.json",
    )

    return parser.parse_args()

def main(args):
    gen_data_paths = glob.glob(os.path.join(args.gen_dataset_path, "gen_data_*.json"))
    # gen_data_paths = glob.glob(os.path.join(args.gen_dataset_path, "gen_data_w_score_*.json"))

    org_data = json.load(open(args.org_dataset_path))
    org_data = [_org_data for _org_data in org_data if _org_data["task"] in ["dvc", "slc"]]

    if len(gen_data_paths) < 1:
        raise FileNotFoundError(f"Check the path {args.gen_dataset_path} has right json files.")
    else:
        print("Will load gen data from ", gen_data_paths)

    gen_data = []

    for gen_data_path in gen_data_paths:
        with open(gen_data_path, "r", encoding="utf-8") as f:
            gen_data += json.load(f)

    print("total number of data: ", len(gen_data))
    cnt = 0
    for itm in org_data:
        _gen_data = [
            gen_itm for gen_itm in gen_data
            if str(gen_itm["qid"].split('|')[1]) == str(itm["task"]) and itm['video'] in gen_itm['video']
            ]

        itm["a"] = [[gen_itm["timestamp"], gen_itm["sentence"]] for gen_itm in _gen_data]

        # Sorting timestamps
        itm["a"] = sorted(itm["a"], key=lambda x: (x[0][0], x[0][1]))

        if len(itm["a"]) == 0:
            itm["a"] = None

    print(f"{cnt} videos are skipped.")
    # save_json(gen_data, os.path.join(args.gen_dataset_path, "gen_data_w_score.json"))
    save_json(org_data, os.path.join(args.gen_dataset_path, "etbench_pred.json"))

if __name__ == "__main__":
    args = get_args()
    main(args)
