
from datasets import load_dataset
import json
from tqdm import tqdm
import random

sft_data = []

# process Open-Orca dataset for English instruction data
dataset = load_dataset("Open-Orca/OpenOrca")
print(f'process orca dataset')
for idx in tqdm(range(len(dataset['train']))):
    sft_data.append({
        'text': f"<|im_start|>system\n{dataset['train'][idx]['system_prompt']}<|im_end|>\n" + \
            f"<|im_start|>question\n{dataset['train'][idx]['question']}<|im_end|>\n" + \
            f"<|im_start|>answer\n{dataset['train'][idx]['response']}<|im_end|>\n",
    })


# process Aya dataset for SEA instruction data

# extra English in aya list, but we do not use english instruction data in aya dataset: 
# ['templated_afriqa', 'templated_joke_explaination', 'templated_mintaka', 'templated_seed_instruct', 'templated_soda']

for sub_domain in reversed(['templated_indo_stories', 'templated_nusax_senti', 'templated_thai_scb', 'templated_thai_usembassy', 'templated_thai_wikitionary', 
                            'templated_thai_pos', 'templated_mintaka', 'templated_ntx_llm', 'templated_scirepeval', 'templated_seed_instruct',
                            'templated_soda', 'templated_uner_llm', 'templated_xcsqa', 'templated_xlel_wd',  'templated_wiki_split', 'templated_xwikis']):

    dataset = load_dataset("CohereForAI/aya_collection", sub_domain)
    print(f'process {sub_domain}')
    if 'train' not in dataset:
        print(f'no train in {sub_domain}')
        continue
    for idx in tqdm(range(len(dataset['train']))):
        if dataset['train'][idx]['language'] not in ['ind', 'zsm', 'vie', 'zho', 'tha']:
            continue
        sft_data.append({
            'text': f"<|im_start|>question\n{dataset['train'][idx]['inputs']}<|im_end|>\n" + \
                f"<|im_start|>answer\n{dataset['train'][idx]['targets']}<|im_end|>\n",
        })


print('Shuffling...')
random.shuffle(sft_data)


print('Write sft data...')
with open('dataset/aya_orca_sft_data.jsonl', 'w') as f:
    for ex in tqdm(sft_data):
        json.dump(ex, f, ensure_ascii=False)
        f.write('\n')
