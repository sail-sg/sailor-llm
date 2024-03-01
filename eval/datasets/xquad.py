import os
import json
from datasets import Dataset, DatasetDict
from opencompass.registry import LOAD_DATASET
from .base import BaseDataset


@LOAD_DATASET.register_module()
class XQUADDataset(BaseDataset):
    @staticmethod
    def load(path):
        data = [json.loads(line.strip()) for line in open(path)]
        return Dataset.from_list([{'context': item['context'], 'question': item['question'], 'answer': item['answer']} for item in data])
