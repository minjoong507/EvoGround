# --------------------------------------------------------
# Dense-Captioning Events in Videos Eval
# Copyright (c) 2017 Ranjay Krishna
# Licensed under The MIT License [see LICENSE for details]
# Written by Ranjay Krishna
# Ported to Python 3 by project maintainers
# --------------------------------------------------------

import argparse
import json
import random
import string
import sys
import os

# Support both installed pycocoevalcap and local coco-caption checkout
_coco_caption_dir = os.path.join(os.path.dirname(__file__), '..', '..', 'coco-caption')
if os.path.isdir(_coco_caption_dir) and _coco_caption_dir not in sys.path:
    sys.path.insert(0, os.path.abspath(_coco_caption_dir))

# Ensure sibling modules (soda, datasets, utils) are importable
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pycocoevalcap.tokenizer.ptbtokenizer import PTBTokenizer
from pycocoevalcap.bleu.bleu import Bleu
from pycocoevalcap.meteor.meteor import Meteor
from pycocoevalcap.rouge.rouge import Rouge
from pycocoevalcap.cider.cider import Cider
import numpy as np


def random_string(string_length):
    letters = string.ascii_lowercase
    return ''.join(random.choice(letters) for _ in range(string_length))


def remove_nonascii(text):
    return ''.join([i if ord(i) < 128 else ' ' for i in text])


class ANETcaptions(object):
    PREDICTION_FIELDS = ['results', 'version', 'external_data']

    def __init__(self, ground_truth_filenames=None, prediction_filename=None,
                 tious=None, max_proposals=1000,
                 prediction_fields=PREDICTION_FIELDS, verbose=False):
        if not tious:
            raise IOError('Please input a valid tIoU.')
        if not ground_truth_filenames:
            raise IOError('Please input a valid ground truth file.')
        if not prediction_filename:
            raise IOError('Please input a valid prediction file.')

        self.verbose = verbose
        self.tious = tious
        self.max_proposals = max_proposals
        self.pred_fields = prediction_fields
        self.ground_truths = self.import_ground_truths(ground_truth_filenames)
        self.prediction = self.import_prediction(prediction_filename)
        self.tokenizer = PTBTokenizer()

        if self.verbose:
            self.scorers = [
                (Bleu(4), ["Bleu_1", "Bleu_2", "Bleu_3", "Bleu_4"]),
                (Meteor(), "METEOR"),
                (Rouge(), "ROUGE_L"),
                (Cider(), "CIDEr"),
            ]
        else:
            self.scorers = [(Meteor(), "METEOR")]

    def import_prediction(self, prediction_filename):
        if self.verbose:
            print("| Loading submission...")
        submission = json.load(open(prediction_filename))
        if not all(field in submission for field in self.pred_fields):
            raise IOError('Submission file is missing required fields: %s' % self.pred_fields)
        results = {}
        for vid_id in submission['results']:
            results[vid_id] = submission['results'][vid_id][:self.max_proposals]
        return results

    def import_ground_truths(self, filenames):
        gts = []
        self.n_ref_vids = set()
        for filename in filenames:
            gt = json.load(open(filename))
            self.n_ref_vids.update(gt.keys())
            gts.append(gt)
        if self.verbose:
            print("| Loading GT. #files: %d, #videos: %d" % (len(filenames), len(self.n_ref_vids)))
        return gts

    def iou(self, interval_1, interval_2):
        start_i, end_i = interval_1[0], interval_1[1]
        start, end = interval_2[0], interval_2[1]
        intersection = max(0, min(end, end_i) - max(start, start_i))
        union = min(max(end, end_i) - min(start, start_i), end - start + end_i - start_i)
        return float(intersection) / (union + 1e-8)

    def check_gt_exists(self, vid_id):
        return any(vid_id in gt for gt in self.ground_truths)

    def get_gt_vid_ids(self):
        vid_ids = set()
        for gt in self.ground_truths:
            vid_ids |= set(gt.keys())
        return list(vid_ids)

    def evaluate(self):
        self.scores = {}
        for tiou in self.tious:
            scores = self.evaluate_tiou(tiou)
            for metric, score in scores.items():
                self.scores.setdefault(metric, []).append(score)
        if self.verbose:
            self.scores['Recall'] = []
            self.scores['Precision'] = []
            for tiou in self.tious:
                precision, recall = self.evaluate_detection(tiou)
                self.scores['Recall'].append(recall)
                self.scores['Precision'].append(precision)
        self.scores['SODA_c'] = [self.compute_soda_c()]

    def compute_soda_c(self):
        """Delegate to the SODA class in soda.py (SODA_c with METEOR scorer)."""
        from soda import SODA
        from datasets import ANETCaptions

        gt_vids = [v for v in self.get_gt_vid_ids() if v in self.prediction]

        # ANETCaptions expects preds as {vid_id: [{"timestamp":…, "sentence":…}]}
        # which is exactly self.prediction's format — pass a shallow copy so
        # preprocess() can replace each vid's entry without mutating self.prediction.
        preds_copy = dict(self.prediction)

        # Deep-copy GT sentences so preprocess() tokenisation doesn't mutate
        # self.ground_truths (which is still needed by evaluate_tiou callers).
        import copy
        gts_copy = copy.deepcopy(self.ground_truths)

        data = ANETCaptions(preds=preds_copy, gts=gts_copy, gt_vid=gt_vids,
                            verbose=self.verbose)
        data.preprocess()

        soda_eval = SODA(data, soda_type="c", tious=[0.0], scorer="Meteor",
                         verbose=self.verbose)
        result = soda_eval.evaluate()
        # result["Meteor"] = [mean_precision, mean_recall, mean_f1]
        return float(result["Meteor"][2])

    def evaluate_detection(self, tiou):
        gt_vid_ids = self.get_gt_vid_ids()
        gt_vid_ids = [v for v in gt_vid_ids if v in self.prediction]
        recall = [0] * len(gt_vid_ids)
        precision = [0] * len(gt_vid_ids)
        for vid_i, vid_id in enumerate(gt_vid_ids):
            best_recall = 0
            best_precision = 0
            for gt in self.ground_truths:
                if vid_id not in gt:
                    continue
                refs = gt[vid_id]
                ref_set_covered = set()
                pred_set_covered = set()
                pred_i = -1
                if vid_id in self.prediction:
                    for pred_i, pred in enumerate(self.prediction[vid_id]):
                        pred_timestamp = pred['timestamp']
                        for ref_i, ref_timestamp in enumerate(refs['timestamps']):
                            if self.iou(pred_timestamp, ref_timestamp) > tiou:
                                ref_set_covered.add(ref_i)
                                pred_set_covered.add(pred_i)
                    if pred_i >= 0:
                        new_precision = float(len(pred_set_covered)) / (pred_i + 1)
                        best_precision = max(best_precision, new_precision)
                new_recall = float(len(ref_set_covered)) / len(refs['timestamps'])
                best_recall = max(best_recall, new_recall)
            recall[vid_i] = best_recall
            precision[vid_i] = best_precision
        return sum(precision) / len(precision), sum(recall) / len(recall)

    def evaluate_tiou(self, tiou):
        res = {}
        gts = {}
        gt_vid_ids = self.get_gt_vid_ids()

        unique_index = 0
        vid2capid = {}
        cur_res = {}
        cur_gts = {}

        for vid_id in gt_vid_ids:
            vid2capid[vid_id] = []

            if vid_id not in self.prediction:
                continue

            for pred in self.prediction[vid_id]:
                has_added = False
                for gt in self.ground_truths:
                    if vid_id not in gt:
                        continue
                    gt_captions = gt[vid_id]
                    for caption_idx, caption_timestamp in enumerate(gt_captions['timestamps']):
                        if self.iou(pred['timestamp'], caption_timestamp) >= tiou:
                            cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                            cur_gts[unique_index] = [{'caption': remove_nonascii(gt_captions['sentences'][caption_idx])}]
                            vid2capid[vid_id].append(unique_index)
                            unique_index += 1
                            has_added = True

                if not has_added:
                    cur_res[unique_index] = [{'caption': remove_nonascii(pred['sentence'])}]
                    cur_gts[unique_index] = [{'caption': random_string(random.randint(10, 20))}]
                    vid2capid[vid_id].append(unique_index)
                    unique_index += 1

        output = {}
        for scorer, method in self.scorers:
            if self.verbose:
                method_name = method if isinstance(method, str) else method[0].split('_')[0]
                print('computing %s score...' % method_name)

            all_scores = {}

            tokenize_res = self.tokenizer.tokenize(cur_res)
            tokenize_gts = self.tokenizer.tokenize(cur_gts)

            for vid in vid2capid:
                res[vid] = {index: tokenize_res[index] for index in vid2capid[vid] if index in tokenize_res}
                gts[vid] = {index: tokenize_gts[index] for index in vid2capid[vid] if index in tokenize_gts}

            for vid_id in gt_vid_ids:
                if vid_id not in self.prediction:
                    continue  # video not processed — exclude from average
                if not res.get(vid_id) or not gts.get(vid_id):
                    all_scores[vid_id] = [0] * len(method) if isinstance(method, list) else 0
                else:
                    score, _ = scorer.compute_score(gts[vid_id], res[vid_id])
                    all_scores[vid_id] = score

            if isinstance(method, list):
                scores = np.mean(list(all_scores.values()), axis=0)
                for m, metric in enumerate(method):
                    output[metric] = scores[m]
                    if self.verbose:
                        print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, metric, output[metric]))
            else:
                output[method] = np.mean(list(all_scores.values()))
                if self.verbose:
                    print("Calculated tIoU: %1.1f, %s: %0.3f" % (tiou, method, output[method]))

        return output


def main(args):
    evaluator = ANETcaptions(
        ground_truth_filenames=args.references,
        prediction_filename=args.submission,
        tious=args.tious,
        max_proposals=args.max_proposals_per_video,
        verbose=args.verbose,
    )
    evaluator.evaluate()

    # SODA_c is a single score (no tIoU sweep); keep it out of per-tIoU table
    tiou_metrics = {k: v for k, v in evaluator.scores.items() if k != 'SODA_c'}

    if args.verbose:
        for i, tiou in enumerate(args.tious):
            print('-' * 80)
            print("tIoU: ", tiou)
            print('-' * 80)
            for metric, scores in tiou_metrics.items():
                print('| %s: %2.4f' % (metric, 100 * scores[i]))

    print('-' * 80)
    print("Average across all tIoUs")
    print('-' * 80)
    for metric, scores in tiou_metrics.items():
        print('| %s: %2.4f' % (metric, 100 * sum(scores) / float(len(scores))))

    print('-' * 80)
    soda_c = evaluator.scores['SODA_c'][0]
    print('| SODA_c: %2.4f' % (100 * soda_c))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate the results stored in a submissions file.')
    parser.add_argument('-s', '--submission', type=str, default='sample_submission.json',
                        help='submission file for ActivityNet Captions evaluation.')
    parser.add_argument('-r', '--references', type=str, nargs='+',
                        default=['dataset/activitynet/val_1.json', 'dataset/activitynet/val_2.json'],
                        help='ground truth reference files.')
    parser.add_argument('--tious', type=float, nargs='+', default=[0.3, 0.5, 0.7, 0.9],
                        help='tIoU thresholds to average over.')
    parser.add_argument('-ppv', '--max-proposals-per-video', type=int, default=1000,
                        help='maximum proposals per video.')
    parser.add_argument('-v', '--verbose', action='store_true',
                        help='Print per-tIoU and all metrics.')
    args = parser.parse_args()
    main(args)
