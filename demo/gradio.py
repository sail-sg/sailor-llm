import gradio as gr
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, StoppingCriteria, StoppingCriteriaList, TextIteratorStreamer

from threading import Thread


model_path = 'sail/Sailor-4B'


# Loading the tokenizer and model from Hugging Face's model hub.
tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True)


# using CUDA for an optimal experience
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model = model.to(device)


# Defining a custom stopping criteria class for the model's text generation.
class StopOnTokens(StoppingCriteria):
    def __call__(self, input_ids: torch.LongTensor, scores: torch.FloatTensor, **kwargs) -> bool:
        stop_ids = [151645]  # IDs of tokens where the generation should stop.
        for stop_id in stop_ids:
            if input_ids[0][-1] == stop_id:  # Checking if the last generated token is a stop token.
                return True
        return False

# user_role = '<|user|>'
# assistant_role = "<|assistant|>"
user_role = 'question'
assistant_role = "answer"

sft_start_token =  "<|im_start|>"
sft_end_token = "<|im_end|>"
ct_end_token = "<|endoftext|>"
# if 'Qwen/' in model_path:
#     sft_end_token = ct_end_token

# Function to generate model predictions.
def predict(message, history):
    # history = []
    history_transformer_format = history + [[message, ""]]
    stop = StopOnTokens()

    # Formatting the input for the model.
    messages = sft_end_token.join([sft_end_token.join([f"\n{sft_start_token}{user_role}:" + item[0], f"\n{sft_start_token}{assistant_role}:" + item[1]])
                        for item in history_transformer_format])
    model_inputs = tokenizer([messages], return_tensors="pt").to(device)
    streamer = TextIteratorStreamer(tokenizer, timeout=10., skip_prompt=True, skip_special_tokens=True)
    generate_kwargs = dict(
        model_inputs,
        streamer=streamer,
        max_new_tokens=256,
        do_sample=True,
        top_p= 0.75,
        top_k= 60,
        temperature=0.2,
        num_beams=1,
        stopping_criteria=StoppingCriteriaList([stop]),
        repetition_penalty=1.1,
    )
    t = Thread(target=model.generate, kwargs=generate_kwargs)
    t.start()  # Starting the generation in a separate thread.
    partial_message = ""
    for new_token in streamer:
        partial_message += new_token
        if sft_end_token in partial_message:  # Breaking the loop if the stop token is generated.
            break
        yield partial_message


# Setting up the Gradio chat interface.
gr.ChatInterface(predict,
                 title=f"Sailor chat",
                 description="Ask Sailor any questions",
                 examples=['How to cook a fish?', 'Who is the president of US now?']
                 ).launch(share=True)  # Launching the web interface.
