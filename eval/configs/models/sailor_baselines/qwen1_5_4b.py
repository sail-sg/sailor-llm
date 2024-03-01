from opencompass.models import HuggingFaceCausalLM

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-4b-hf',
        path="Qwen/Qwen1.5-4B",
        tokenizer_path='Qwen/Qwen1.5-4B',
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]
