#!/bin/bash
cd sailor-llm

# setup opencompass environment
conda create --name opencompass python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate opencompass
git clone https://github.com/open-compass/opencompass opencompass
cd opencompass
pip install -e .
pip install pythainlp langid
mkdir data

# create evaluation scripts
cd ../
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

# run evaluation
cd opencompass
python run.py configs/eval_sailor.py -w outputs/sailor --num-gpus 4 --max-num-workers 64

