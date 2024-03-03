## How to Fine-tune Sailor

You can try this code to fine-tune data you want. The script supports the training with [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory)

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

git clone https://github.com/hiyouga/LLaMA-Factory.git
cd LLaMA-Factory

# checking out the specific commit hash (Optional)
git checkout 4e5fae2fac85227641bd16159cf296a32e0b18b4

# install the required packages
pip install -r requirements.txt
```

#### Set chat template
Insert the code below in `src/llmtuner/data/template.py` for registering Sailor template. You can also use the other templates or design your own template.
```python
_register_template(
    name="sailor",
    format_user=StringFormatter(slots=["<|im_start|>question\n{{content}}<|im_end|>\n<|im_start|>answer\n"]),
    format_system=StringFormatter(slots=["<|im_start|>system\n{{content}}<|im_end|>\n"]),
    format_separator=EmptyFormatter(slots=["\n"]),
    default_system="You are a helpful assistant.",
    stop_words=["<|im_end|>"],
    replace_eos=True,
)
```

#### Training
##### Training with fsdp config
Here we provide an example for fine-tuning the model with fsdp under 8 GPUs cards (`num_machines` means the number of GPUs you want to train the model with) with the config below:
```yaml
compute_environment: LOCAL_MACHINE
distributed_type: FSDP
downcast_bf16: 'no'
fsdp_config:
  fsdp_auto_wrap_policy: TRANSFORMER_BASED_WRAP
  fsdp_backward_prefetch_policy: BACKWARD_PRE
  fsdp_cpu_ram_efficient_loading: true
  fsdp_forward_prefetch: false
  fsdp_offload_params: false
  fsdp_sharding_strategy: 1
  fsdp_state_dict_type: FULL_STATE_DICT
  fsdp_sync_module_states: true
  fsdp_use_orig_params: true
machine_rank: 0
main_training_function: main
mixed_precision: bf16
num_machines: 1
num_processes: 8
rdzv_backend: static
same_network: true
tpu_env: []
tpu_use_cluster: false
tpu_use_sudo: false
use_cpu: false
```

##### Runing supervised fine-tuning
```bash
# copy training data and scripts
cp -r ../dataset/ .
cp ../fsdp_config.yaml .
cp ../run.sh .

# start finetuning
bash run.sh
```
