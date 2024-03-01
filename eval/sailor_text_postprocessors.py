import re
from typing import Callable, Optional, Union
from opencompass.registry import TEXT_POSTPROCESSORS


@TEXT_POSTPROCESSORS.register_module('general_pred')
def general_pred_postprocess(text: str) -> str:
    # Cut off the first newline
    text = re.split(r'[\n]', text, 1)[0]

    # Remove blank spaces before words
    text = re.sub(r'^\s*', '', text)

    prompts = set([
        "คำตอบ: ".lower(),
        "Trả lời: ".lower(),
        "Jawaban: ".lower(),
        "ไทย: ".lower(),
        "Bahasa Indonesia: ".lower(),
        "Tiếng Việt: ".lower(),
    ])
    for prompt in prompts:
        if prompt in text:
            text = text.split(prompt)[-1]
            break

    # Remove punctuation after the text
    no_punctuation = re.sub(r'(\w)[^\w\s]*\s*$', r'\1', text)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', no_punctuation)

    # Remove blank spaces before comma
    cleaned_text = re.sub(r'\s*,', ',', cleaned_text)

    # Remove blank spaces before period
    cleaned_text = re.sub(r'\s*\.', '.', cleaned_text)

    return cleaned_text.strip()


@TEXT_POSTPROCESSORS.register_module('general_ans')
def general_ans_postprocess(text: str) -> str:
    # Remove blank spaces before words
    text = re.sub(r'^\s*', '', text)

    # Remove punctuation after the text
    text = re.sub(r'(\w)[^\w\s]*\s*$', r'\1', text)

    # Remove duplicated blank spaces
    cleaned_text = re.sub(r'\s+', ' ', text)

    # Remove blank spaces before comma
    cleaned_text = re.sub(r'\s*,', ',', cleaned_text)

    # Remove blank spaces before period
    cleaned_text = re.sub(r'\s*\.', '.', cleaned_text)

    return cleaned_text.strip()



def first_option_parse(text: str, options: str, cushion=True) -> str:
    """Find first valid option for text."""

    # yapf: disable
    # flake8: noqa: W605
    patterns = [
        f'答案是?\s?([{options}])',
        f'答案是?\s?：([{options}])',
        f'答案是?\s?:([{options}])',
        f'答案应该?是\s?([{options}])',
        f'答案应该?选\s?([{options}])',
        f'答案为\s?([{options}])',
        f'答案选\s?([{options}])',
        f'选择?\s?([{options}])',
        f'只有选?项?\s?([{options}])\s?是?对',
        f'只有选?项?\s?([{options}])\s?是?错',
        f'只有选?项?\s?([{options}])\s?不?正确',
        f'只有选?项?\s?([{options}])\s?错误',
        f'说法不?对选?项?的?是\s?([{options}])',
        f'说法不?正确选?项?的?是\s?([{options}])',
        f'说法错误选?项?的?是\s?([{options}])',
        f'([{options}])\s?是正确的',
        f'([{options}])\s?是正确答案',
        f'选项\s?([{options}])\s?正确',
        f'所以答\s?([{options}])',
        f'1.\s?([{options}])[.。$]?',
        f'[\s，：:,]([{options}])[。，,\.]?',
        f'[\s，,：:][故即]([{options}])[。\.]?',
        f'[\s，,：:]因此([{options}])[。\.]?',
        f'[是为。]\s?([{options}])[。\.]?',
        f'因此\s?([{options}])[。\.]?',
        f'显然\s?([{options}])[。\.]?',
        f'1.\s?(.*?)',
        f'1.\s?([{options}])[.。$]?$',
        f'所以\s?([{options}][.。$]?$)',
        f'所有\s?([{options}][.。$]?$)',
        f'[\s，：:,]([{options}])[。，,\.]?$',
        f'[\s，,：:][故即]([{options}])[。\.]?$',
        f'[\s，,：:]因此([{options}])[。\.]?$',
        f'[是为。]\s?([{options}])[。\.]?$',
        f'因此\s?([{options}])[。\.]?$',
        f'显然\s?([{options}])[。\.]?$',
        f'答案是\s?(\S+)(?:。|$)',
        f'答案应该是\s?(\S+)(?:。|$)',
        f'答案为\s?(\S+)(?:。|$)',
        f'[Tt]he answer is ([{options}])',
        f'[Tt]he answer is option ([{options}])',
        f'[Tt]he correct answer is ([{options}])',
        f'[Tt]he correct answer is option ([{options}])',
        f'[Tt]he answer to the question is ([{options}])',
        f'^选项\s?([{options}])',
        f'^([{options}])\s?选?项',
        f'(\s|^)[{options}][\s。，,：:\.$]',
        f'(\s|^)[{options}](\s|$)',
        f'1.\s?(.*?)$',
        f'1.\s?([{options}])[.。$]?$',
    ]
    cushion_patterns = [
        f'([{options}]):',
        f'[{options}]',
    ]
    # flake8: noqa
    # yapf: enable

    if cushion:
        patterns.extend(cushion_patterns)
    for pattern in patterns:
        match = re.search(pattern, text)
        if match:
            outputs = match.group(0)
            for i in options:
                if i in outputs:
                    return i
    return ''