import os
import langid
import random
import evaluate
import numpy as np
from typing import List
from collections import Counter
from pythainlp.tokenize import word_tokenize
from .icl_base_evaluator import BaseEvaluator
from .icl_hf_evaluator import HuggingfaceEvaluator
from opencompass.registry import ICL_EVALUATORS
from opencompass.utils import general_ans_postprocess, general_pred_postprocess




@ICL_EVALUATORS.register_module()
class AnsEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        em_cnt = 0
        f1_cnt = 0
        details = []
        for pred, ans in zip(predictions, references):
            if not isinstance(pred, str):
                pred = str(pred)
            if not isinstance(ans, str):
                ans = str(ans) 

            processed_pred = general_pred_postprocess(pred.lower())
            processed_ans = general_ans_postprocess(ans.lower())

            pred_set = set([
                pred.lower(),
                processed_pred
            ])  # noqa
            ans_set = set([
                ans.lower(),
                processed_ans
            ])  # noqa
            detail = {'pred': pred, 'answer': ans}
            if len(pred_set & ans_set) > 0:
                em_cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

            lang, _ = langid.classify(pred)
            if lang == 'th':
                pred_words = word_tokenize(processed_pred, join_broken_num=True, keep_whitespace=False)
                ans_words = word_tokenize(processed_ans, join_broken_num=True, keep_whitespace=False)
            else:
                pred_words = processed_pred.split()
                ans_words = processed_ans.split()

            common_words = Counter(pred_words) & Counter(ans_words)
            common_num = sum(common_words.values())

            if len(common_words) == 0:
                f1 = 0
            else:
                prec = 1.0 * common_num / len(pred_words) if len(pred_words) != 0 else 0
                rec = 1.0 * common_num / len(ans_words) if len(ans_words) != 0 else 0
                f1 = 2 * (prec * rec) / (prec + rec) if prec + rec != 0 else 0
            f1_cnt += f1



        em_score = em_cnt / len(predictions) * 100
        f1_score = f1_cnt / len(predictions) * 100

        return {'EM': em_score, 'F1': f1_score, 'details': details}




class TextGenEvaluator(BaseEvaluator):
    def __init__(self, seed: int = 0) -> None:
        self.seed = seed
        super().__init__()

    def _preprocess(self, predictions: List, references: List) -> dict:
        """Preprocess the final predictions and references to needed format.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: preprocessed results.
        """
        return {
            'predictions': [general_pred_postprocess(pred.lower()) for pred in predictions],
            'references': [ref.lower() for ref in references],
        }

    def _postprocess(self, bleu_results: dict, chrfpp_results: dict) -> dict:
        """Postprocess for final scores.

        Args:
            scores (dict): Dict of calculated scores of metrics.

        Returns:
            dict: postprocessed scores.
        """

        return {'BLEU': bleu_results['score'], 'Chrf++': chrfpp_results['score']}

    def score(self, predictions: List, references: List) -> dict:
        """Calculate scores.

        Args:
            predictions (List): List of predictions of each sample.
            references (List): List of targets for each sample.

        Returns:
            dict: calculated scores.
        """
        random_state = random.getstate()
        np_random_state = np.random.get_state()

        random.seed(self.seed)
        np.random.seed(self.seed)
        if len(predictions) != len(references):
            return {
                'error':
                'predictions and references have different '
                f'length. len(predictions): {len(predictions)}, '
                f'len(references): {len(references)}'
            }

        self.metric = 'sacrebleu'
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)
        bleu_results = metric.compute(**self._preprocess(predictions, references))


        self.metric = 'chrf'
        local_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                                  'hf_metrics', self.metric + '.py')
        if os.path.exists(local_path):
            metric = evaluate.load(local_path)
        else:
            metric = evaluate.load(self.metric)
        chrfpp_results = metric.compute(**self._preprocess(predictions, references), word_order=2)

        result = self._postprocess(bleu_results, chrfpp_results)
        random.setstate(random_state)
        np.random.set_state(np_random_state)
        return result




