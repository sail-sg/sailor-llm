import os
import re
import json
from datasets import Dataset
from opencompass.openicl.icl_evaluator import BaseEvaluator
from opencompass.registry import ICL_EVALUATORS, LOAD_DATASET
from opencompass.utils import first_option_parse

from .base import BaseDataset


@LOAD_DATASET.register_module()
class M3EXAMTHDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = Dataset.from_list(json.load(open(path)))

        def preprocess(example):
            trans = {
                '๕': '5',
                '๔': '4',
                '๓': '3',
                '๒': '2',
                '๑': '1',
                '5': '๕',
                '4': '๔',
                '3': '๓',
                '2': '๒',
                '1': '๑'
            }

            example['full_options'] = '\n'.join(example['options'])

            if '0' <= example['answer_text'] <= '9':
                example['answer'] = example['answer_text']
            else:
                example['answer'] = trans[example['answer_text']]

            for idx in range(len(example['options'])):
                example[f'options_{idx}'] = example['options'][idx][2:].lstrip()      

            return example

        return dataset.map(preprocess)




@LOAD_DATASET.register_module()
class M3EXAMVIDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = Dataset.from_list(json.load(open(path)))
        def preprocess(example):
            if ord(example['answer_text']) == 1042:
                 example['answer_text'] = 'B'
            for idx in range(len(example['options'])):
                if ord(example['options'][idx][0]) == 1042:
                    example['options'][idx] = list(example['options'][idx])
                    example['options'][idx][0] = 'B'
                    example['options'][idx] = "".join(example['options'][idx])
                
                example[f'options_{idx}'] = example['options'][idx][2:].lstrip()
                if example[f'options_{idx}'][0] == '.':
                    example[f'options_{idx}'] = example[f'options_{idx}'][1:]

            example['full_options'] = '\n'.join(example['options'])
     
            return example

        return dataset.map(preprocess)




@LOAD_DATASET.register_module()
class M3EXAMJvDataset(BaseDataset):

    @staticmethod
    def load(path):
        dataset = Dataset.from_list(json.load(open(path)))
        def preprocess(example):
            example['question_text'] = example['question_text'].rstrip('. …')
            example['answer_text'] = example['answer_text'].upper()
            for idx in range(len(example['options'])):
                example['options'][idx] = list(example['options'][idx])
                example['options'][idx][0] = example['options'][idx][0].upper()
                example['options'][idx] = "".join(example['options'][idx])

                example[f'options_{idx}'] = example['options'][idx][2:].lstrip()  

            example['full_options'] = '\n'.join(example['options'])       
            return example

        return dataset.map(preprocess)



@ICL_EVALUATORS.register_module()
class M3examEmEvaluator(BaseEvaluator):
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
            "Câu hỏi: "
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



@ICL_EVALUATORS.register_module()
class M3examThEvaluator(BaseEvaluator):
    """Exact match evaluator."""

    def __init__(self) -> None:
        super().__init__()

    def score(self, predictions, references):
        trans = {
            '๕': '5',
            '๔': '4',
            '๓': '3',
            '๒': '2',
            '๑': '1',
            '5': '๕',
            '4': '๔',
            '3': '๓',
            '2': '๒',
            '1': '๑'
        }

        if len(predictions) != len(references):
            return {
                'error': 'predictions and references have different '
                'length'
            }

        cnt = 0
        details = []
        for pred, ans in zip(predictions, references):
            detail = {'pred': pred, 'answer': ans}
            # Cut off the first newline
            pred = re.split(r'[\n]', pred, 1)[0]

            pred = first_option_parse(pred, '12345๑๒๓๔๕')

            if pred == ans or trans[ans] == pred:
                cnt += 1
                detail['correct'] = True
            else:
                detail['correct'] = False
            details.append(detail)

        score = cnt / len(predictions) * 100

        return {'EM': score, 'details': details}
