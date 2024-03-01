import os
import json
from collections import Counter
from datasets import Dataset, DatasetDict
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from .base import BaseDataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.utils import general_ans_postprocess, general_pred_postprocess


@LOAD_DATASET.register_module()
class TydiqaIdDataset(BaseDataset):
    @staticmethod
    def load(path):
        return Dataset.from_list([json.loads(line.strip()) for line in open(path)])

class TydiqaIdEvaluator(BaseEvaluator):
    # This evaluation class is edited from:
    #  https://github.com/allenai/bi-att-flow/blob/master/squad/evaluate-v1.1.py
    def f1_score(self, prediction, ground_truth):
        prediction_tokens = general_pred_postprocess(prediction.lower()).split()
        ground_truth_tokens = general_ans_postprocess(ground_truth.lower()).split()
        common = Counter(prediction_tokens) & Counter(ground_truth_tokens)
        num_same = sum(common.values())
        if num_same == 0:
            return 0
        precision = 1.0 * num_same / len(prediction_tokens)
        recall = 1.0 * num_same / len(ground_truth_tokens)
        f1 = (2 * precision * recall) / (precision + recall)
        return f1

    def exact_match_score(self, prediction, ground_truth):
        return (general_pred_postprocess(prediction.lower()) == general_ans_postprocess(
            ground_truth.lower()))

    def metric_max_over_ground_truths(self, metric_fn, prediction,
                                      ground_truths):
        scores_for_ground_truths = []
        for ground_truth in ground_truths:
            score = metric_fn(prediction, ground_truth)
            scores_for_ground_truths.append(score)
        return max(scores_for_ground_truths)

    def score(self, predictions, references):
        f1 = exact_match = total = 0
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }
        for prediction, reference in zip(predictions, references):
            exact_match += self.metric_max_over_ground_truths(
                self.exact_match_score, prediction, reference)
            f1 += self.metric_max_over_ground_truths(self.f1_score, prediction,
                                                     reference)
            total += 1

        exact_match = 100.0 * exact_match / total
        f1 = 100.0 * f1 / total

        return {'EM': exact_match, 'F1': f1}