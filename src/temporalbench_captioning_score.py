import json
from tqdm import tqdm
import argparse
import os
import glob

from sentence_transformers import SentenceTransformer, util
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.cider.cider import Cider
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer


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


def calculate_average_similarity(ref_list, gt_list, model_name='all-MiniLM-L6-v2'):
    # Initialize the model and move to CUDA
    model = SentenceTransformer(model_name)
    model = model.to('cuda')



    # Combine ref and gt lists into a big batch for encoding
    combined_sentences = ref_list + gt_list

    # Encode the batch with CUDA
    embeddings = model.encode(combined_sentences, convert_to_tensor=True, device='cuda')

    # Split embeddings into ref and gt parts
    ref_embeddings = embeddings[:len(ref_list)]
    gt_embeddings = embeddings[len(ref_list):]

    # Calculate cosine similarities between each ref and gt pair
    cosine_scores = util.cos_sim(ref_embeddings, gt_embeddings).diagonal()

    # Calculate the average similarity
    avg_similarity = cosine_scores.mean().item()

    return avg_similarity

def evaluate(gt_captions, pred_captions):

    tokenizer = PTBTokenizer()

    scorers = [
        (Bleu(4), ['Bleu_1', 'Bleu_2', 'Bleu_3', 'Bleu_4']),
        (Meteor(), 'METEOR'),
        (Rouge(), 'ROUGE_L'),
        (Cider(), 'CIDEr')
    ]

    gt_dict = {}
    pred_dict = {}

    for i, (gt, pred) in enumerate(zip(gt_captions, pred_captions)):
        gt_dict[i] = [{'caption': gt}]
        pred_dict[i] = [{'caption': pred}]

    # tokenize captions
    gt_dict = tokenizer.tokenize(gt_dict)
    pred_dict = tokenizer.tokenize(pred_dict)

    results = {}

    for scorer, method in scorers:

        if isinstance(method, list):
            score, scores = scorer.compute_score(gt_dict, pred_dict)

            for m, s in zip(method, score):
                results[m] = s

        else:
            score, scores = scorer.compute_score(gt_dict, pred_dict)
            results[method] = score

    return results


if __name__ == '__main__':
    # Create an ArgumentParser object
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument('--data_folder', type=str, default="dataset/temporalbench/",
                        help='Path to dataset (from Huggingface)')
    parser.add_argument("--gen_dataset_path", type=str, required=True, help="Output directory of score files")
    # parser.add_argument("--nframes", type=int, default=1, help="Number of frames to sample.")

    # Parse arguments
    args = parser.parse_args()

    with open(os.path.join(args.data_folder, 'temporalbench_short_caption.json'), 'r') as f:
        gt_data = json.load(f)

    print("========== Total gt data:", len(gt_data))

    gen_data_paths = glob.glob(os.path.join(args.gen_dataset_path, "gen_data_*.json"))
    gen_data = []

    for gen_data_path in gen_data_paths:
        gen_data += json.load(open(gen_data_path))

    print("========== Total prediction data:", len(gen_data))

    gt_captions = []
    pred_captions = []
    cmnt = 0
    for gt_itm in gt_data:
        find = False

        for gen_itm in gen_data:
            if gt_itm['idx'] in gen_itm['qid']:
                gt_captions.append(gt_itm['GT'])
                pred_captions.append(gen_itm['response'])
                fine = True
                break

        if find is False:
            cmnt+=1
            # print(gt_itm)
    print(cmnt)
    print(f"total {len(gt_captions)} caption pairs will be evaluated.")

    similarity = calculate_average_similarity(pred_captions, gt_captions)
    metrics = evaluate(gt_captions, pred_captions)

    metrics['similarity'] = similarity
    for k, v in metrics.items():
        metrics[k] = round(v * 100, 4)
    print('Results:', metrics)

    output_path = os.path.join(args.gen_dataset_path, "metrics.json")

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=4, ensure_ascii=False)
        print(f"Save to: {output_path}")

    save_json(gen_data, os.path.join(args.gen_dataset_path, "gen_data.json"))





