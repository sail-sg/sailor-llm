from opencompass.models import HuggingFaceCausalLM


_meta_template = dict(
    round=[
        dict(role="HUMAN", begin='<|im_start|>user\n', end='<|im_end|>\n'),
        dict(role="BOT", begin="<|im_start|>assistant\n", end='<|im_end|>\n', generate=True),
    ],
    reserved_roles=[dict(role='SYSTEM', begin='<|im_start|>system\n', end='<|im_end|>\n'),],
    eos_token_id=151645,
)

models = [
    dict(
        type=HuggingFaceCausalLM,
        abbr='qwen1.5-7b-chat-hf',
        path="Qwen/Qwen1.5-7B-Chat",
        model_kwargs=dict(
            device_map='auto',
            trust_remote_code=True
        ),
        generation_kwargs=dict(top_k=20,
            top_p=0.8,
            temperature=0.7,
            repetition_penalty=1.05
        ),
        tokenizer_kwargs=dict(
            padding_side='left',
            truncation_side='left',
            trust_remote_code=True,
            use_fast=False,
        ),
        meta_template=_meta_template,
        max_out_len=100,
        max_seq_len=4096,
        batch_size=8,
        batch_padding=True,
        run_cfg=dict(num_gpus=1, num_procs=1),
        end_str='<|im_end|>',
    )
]
