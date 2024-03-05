import llama_cpp
import llama_cpp.llama_tokenizer
import gradio as gr

llama = llama_cpp.Llama.from_pretrained(
    repo_id="sail/Sailor-4B-Chat-gguf",
    filename="ggml-model-Q3_K_L.gguf",
    tokenizer=llama_cpp.llama_tokenizer.LlamaHFTokenizer.from_pretrained("sail/Sailor-4B-Chat"),
    n_gpu_layers=40,
    n_threads=8,
    verbose=False,
)

system_role= 'system'
user_role = 'question'
assistant_role = "answer"

sft_start_token =  "<|im_start|>"
sft_end_token = "<|im_end|>"
ct_end_token = "<|endoftext|>"

system_prompt= \
'You are an AI assistant named Sailor created by Sea AI Lab. \
Your answer should be friendly, unbiased, faithful, informative and detailed.'
system_prompt = f"<|im_start|>{system_role}\n{system_prompt}<|im_end|>"

def predict(message, history):
    history_transformer_format = history + [[message, ""]]

    # Formatting the input for the model.
    messages = system_prompt + sft_end_token.join([
        sft_end_token.join([
            f"\n{sft_start_token}{user_role}\n" + item[0],
            f"\n{sft_start_token}{assistant_role}\n" + item[1]
        ]) for item in history_transformer_format
    ])

    response = llama(
        messages,
        stream=True,
        top_p=0.75,
        top_k=60,
        stop=["<|im_end|>", "<|endoftext|>"],
        temperature=0.2,
        max_tokens=256,
    )

    text = ""
    for chunk in response:
        res = chunk["choices"][0]
        
        if "text" not in res:
            continue
        text += res["text"]
        yield text


css = """
full-height {
    height: 100%;
}
"""

prompt_examples = [
    'How to cook a fish?',
    'Cara memanggang ikan',
    'วิธีย่างปลา',
    'Cách nướng cá'
]

with gr.Blocks(theme=gr.themes.Soft(), fill_height=True) as demo:
    gr.Markdown("""<p align="center"><img src="https://github.com/sail-sg/sailor-llm/raw/main/misc/wide_sailor_banner.jpg" style="height: 110px"/><p>""")
    gr.ChatInterface(predict, fill_height=True, examples=prompt_examples, css=css)
    
    demo.launch(share=True)  # Launching the web interface.


