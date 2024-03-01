import re
import json
from datasets import Dataset
from opencompass.registry import LOAD_DATASET, ICL_EVALUATORS
from opencompass.openicl.icl_evaluator import BaseEvaluator
from .base import BaseDataset
from opencompass.utils import first_option_parse


@LOAD_DATASET.register_module()
class BelebeleDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = Dataset.from_list([json.loads(line.strip()) for line in open(path)])

        def preprocess(example):
            example['options'] = '\n'.join([f'{chr(64+idx)}. ' + example[f'mc_answer{idx}'] for idx in range(1, 5)])  
            example['answer'] = chr(int(example['correct_answer_num'])+64)   
            return example

        return dataset.map(preprocess)



@ICL_EVALUATORS.register_module()
class BelebeleEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        cnt = 0
        details = []
        prompts = set([
            "Context: ",
            "Answer: ",
            "Câu hỏi: ",
            "Bối cảnh: "
        ])
        for pred, ans in zip(predictions, references):
            detail = {'pred': pred, 'answer': ans}

            # Cut off the first newline
            pred = re.split(r'[\n]', pred, 1)[0]

            for prompt in prompts:
                if prompt in pred:
                    pred = pred.split(prompt)[-1]
                    break

            pred = first_option_parse(pred, 'ABCD')

            if pred == ans:
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'EM': score, 'details': details}