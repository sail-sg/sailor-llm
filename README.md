# Sailor: Open Language Models for South-East Asia

This repository contains the evaluation and demo code for Sailor, a suite of open language models for South-East Asia. Sailor is developed by the [Sea AI Lab](https://sail.sea.com/) and [Singapore University of Technology and Design](https://istd.sutd.edu.sg/people/faculty/lu-wei/).

<img src="misc/banner.jpg" style="width:90%">

## Introduction

Sailor is a suite of Open Language Models tailored for South-East Asia (SEA), focusing on languages such as 🇮🇩Indonesian, 🇹🇭Thai, 🇻🇳Vietnamese, 🇲🇾Malay, and 🇱🇦Lao. Developed with careful data curation, Sailor models are designed to understand and generate text across diverse linguistic landscapes of SEA region. Built from Qwen 1.5, Sailor encompasses models of varying sizes, spanning from 0.5B to 7B versions for different requirements. Benchmarking results demonstrate Sailor's proficiency in tasks such as question answering, commonsense reasoning, reading comprehension and etc. in SEA languages.

- Continually pretrained on 200 Billion to 400 Billion tokens over 7 languages, including Indonesian, Thai, Vietnamese, Malay, Lao, English and Chinese.
- Various model sizes (0.5B, 1.8B, 4B and 7B) to support different requirements.
- Strong performance on SEA benchmarks such as XQuAD, TydiQA, XCOPA, Belebele and M3Exam.
- No restrict on the research and the commercial use, but should comply with the Qwen 1.5 license.

## Models

You can find all the Sailor models in our Huggingface home page [here](https://huggingface.co/sail):
- [Sailor-0.5B](https://huggingface.co/sail/Sailor-0.5B)
- [Sailor-1.8B](https://huggingface.co/sail/Sailor-1.8B)
- [Sailor-4B](https://huggingface.co/sail/Sailor-4B)
- [Sailor-7B](https://huggingface.co/sail/Sailor-7B)

## Evaluation

You can find the evaluation code to reproduce the results in the `eval` directory. The evaluation results are presented in the form of tables, where the first column is the model name, and the reset columns are the performance on Thai (th), Indonesian (id), and Vietnamese (vi) languages, respectively. The results of Sailor models are highlighted in bold.

### Setup

We use [OpenCompass](https://github.com/open-compass/opencompass) to evaluate the models. To install the required packages, run the following command under this folder:

```bash
# setup opencompass environment
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
pip install pythainlp langid
mkdir data
```

### Build Evaluation Script

To build the evaluation script, run the following command under this folder:

```bash
cp -r eval/configs/* opencompass/configs/
cp -r eval/data/* opencompass/data/
cp -r eval/datasets/* opencompass/opencompass/datasets/
cp eval/icl_sailor_evaluator.py opencompass/opencompass/openicl/icl_evaluator/
cp eval/sailor_text_postprocessors.py opencompass/opencompass/utils/
echo "from .icl_sailor_evaluator import AnsEvaluator, TextGenEvaluator  # noqa" >> "opencompass/opencompass/openicl/icl_evaluator/__init__.py"
echo "from .sailor_text_postprocessors import *  # noqa" >> "opencompass/opencompass/utils/__init__.py"
echo "from .xquad import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .tydiqa_id import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .xcopa_sea import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .m3exam import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
echo "from .belebele import *  # noqa: F401, F403" >> "opencompass/opencompass/datasets/__init__.py"
cp eval/eval_sailor.py opencompass/configs/
```

### Run Evaluation

To run the evaluation, run the following command under this folder:

```bash
cd opencompass
python run.py configs/eval_sailor.py -w outputs/sailor --num-gpus 1 --max-num-workers 64
```

You can also modify the script to evaluate other models like Qwen1.5, Llama, Mistral, etc.


## Demo

We provide a simple demo to chat with Sailor-Chat models (coming soon). You can find the demo code in the `demo` directory.