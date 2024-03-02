## How to Fine-tune Sailor

You can try our code to fine-tune data you want. The script supports the training with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

### Prepare Datasets

Here we give an example to preprocess fine-tune data from [Orca](https://huggingface.co/datasets/Open-Orca/OpenOrca) and [Aya datase](https://huggingface.co/datasets/CohereForAI/aya_collection)

```bash
pip install datasets
python prepare_dataset.py
```

### Training

#### Setup
To install the required packages, run the following command under this folder:

```bash
# create conda environment
conda create --name finetune python=3.10 pytorch torchvision pytorch-cuda -c nvidia -c pytorch -y
conda activate finetune


# install LLaMA-Factory
# commit hash 4e5fae2fac85227641bd16159cf296a32e0b18b4
git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory
pip install -r requirements.txt

```

#### Start finetuning
```bash
# copy training data and scripts
cp -r ../dataset/ .
cp ../fsdp_config.yaml .
cp ../run.sh .

# start finetuning
bash run.sh
```
