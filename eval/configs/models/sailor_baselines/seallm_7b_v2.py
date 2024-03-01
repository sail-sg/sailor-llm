from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='</s>'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='</s>', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<s><|im_start|>system\n', end='</s>'),],
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='seallm_7b_v2',
        path="SeaLLMs/SeaLLM-7B-v2",
        tokenizer_path='SeaLLMs/SeaLLM-7B-v2',
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
        meta_template=_meta_template,
        run_cfg=dict(num_gpus=1, num_procs=1),
    )
]