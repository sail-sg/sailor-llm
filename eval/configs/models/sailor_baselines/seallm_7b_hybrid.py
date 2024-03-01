from opencompass.models import HuggingFaceCausalLM


models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='seallm_7b_hybrid',
        path="SeaLLMs/SeaLLM-7B-Hybrid",
        tokenizer_path='SeaLLMs/SeaLLM-7B-Hybrid',
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            use_fast=False,
        ),
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        model_kwargs=dict(device_map='auto'),
        batch_padding=True, # if false, inference with for-loop without batch padding
        run_cfg=dict(num_gpus=1, num_procs=1)
    )
]
