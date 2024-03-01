
## Evaluation

| QA / 3-shot (EM / F1) | XQuAD (Th) | TyDi QA (Id) | XQuAD (Vi) 
|-----------| -------  | ------- | ------- | 
| [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) | 14.19 / 23.35 |  20.71 / 32.64  | 19.85 / 35.38
| [Sailor-0.5B](https://huggingface.co/sail/Sailor-0.5B)  | **15.84**	/ **27.58**	| **30.44**	/ **54.74**	| **21.13**	/ **40.57**
|-----------| -------  | ------- | ------- |
| [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) | 27.24 / 43.56 | 29.73 / 53.76 | 29.17 / 48.15
| [Sailor-1.8B](https://huggingface.co/sail/Sailor-1.8B)   | **32.72** / **48.66** | **40.88** / **65.37** | **34.22** / **53.35**
|-----------| -------  | ------- | ------- |
| [Qwen1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B)   | 34.03 / 53.40 | 48.32 / 72.68 | 43.71 / 63.86
| [Sailor-4B](https://huggingface.co/sail/Sailor-4B)   | **46.82** / **63.34** | **53.98** / **73.48** | **47.65** / **67.09**
|-----------| -------  | ------- | ------- |
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  | 30.64 / 43.80 | 56.64 / 72.14 | 46.96 / 66.16
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 48.48 / 63.27 | **63.54** / **78.73** | 53.72 / 72.75
| [SeaLLM-7b-Hybrid](https://huggingface.co/SeaLLMs/SeaLLM-7B-Hybrid)  | 49.70 / 67.62 | 50.62 / 75.21 | 49.62 / 70.74
| [SeaLLM-7b-v2](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2)  | 34.55 / 55.13 | 52.21 / 77.00 | 46.19 / 72.11
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)  | 53.79 / 69.30 | 57.17 / 77.28 | **56.63** / **76.99**
| [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) | 18.89 / 41.56 | 29.91 / 52.81 | 31.05 / 56.80
| [Sailor-7B](https://huggingface.co/sail/Sailor-7B) | **57.88** / **71.06** | 60.53 / 75.42 | 53.81 / 74.62





| XCOPA / 3-shot (EM) |  Th | Id | Vi
|-----------| -------  | ------- | ------- | 
| Random    |  <span style="color: gray">50.00</span> | <span style="color: gray">50.00</span> | <span style="color: gray">50.00</span>
| [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) | 51.00 | 52.20 | 53.80 |
| [Sailor-0.5B](https://huggingface.co/sail/Sailor-0.5B)   | 51.00 | **58.20** | **58.00** |
|-----------| -------  | ------- | ------- |
| [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) | 52.60 | 51.60 | 53.40 |
| [Sailor-1.8B](https://huggingface.co/sail/Sailor-1.8B)   | **53.80** | **64.20** | **63.20** |
|-----------| -------  | ------- | ------- |
| [Qwen1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B)   | 53.40 | 55.00 | 57.80 |
| [Sailor-4B](https://huggingface.co/sail/Sailor-4B)   | 53.40 | **69.20** | **68.20** |
|-----------| -------  | ------- | ------- |
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  | 52.80 | 64.00 | 62.00 |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1) | 57.20 | 62.40 | 61.60 |
| [SeaLLM-7b-Hybrid](https://huggingface.co/SeaLLMs/SeaLLM-7B-Hybrid)  | 58.20 | 71.60 | 67.60 | 
| [SeaLLM-7b-v2](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2)  | 56.80 | 64.00 | 64.60 | 
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)  | 54.20 | 62.20 | 66.20 | 
| [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) | 53.80 | 65.20 | 64.40 | 
| [Sailor-7B](https://huggingface.co/sail/Sailor-7B) | **59.00** | **72.20** | **72.20** | 






| M3Exam / 3-shot (EM) | Th | Id | Vi 
|-----------| -------  | ------- | ------- | 
| Random    |  <span style="color: gray">22.90</span> | <span style="color: gray">25.00</span> | <span style="color: gray">25.21</span>
| [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) | 22.93 | 25.07 | 26.66
| [Sailor-0.5B](https://huggingface.co/sail/Sailor-0.5B)   | **24.41** | **26.15** | **30.91** |
|-----------| -------  | ------- | ------- |
| [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) | 24.04 | 24.26 | 28.68
| [Sailor-1.8B](https://huggingface.co/sail/Sailor-1.8B)   | **25.38** | **28.30** | **34.71** |  
|-----------| -------  | ------- | ------- |
| [Qwen1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B)   | 24.50 | 24.26 | 30.02 
| [Sailor-4B](https://huggingface.co/sail/Sailor-4B)   | **27.88** | **31.27** | **40.69** |
|-----------| -------  | ------- | ------- |
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  | 23.67 | 25.07 | 33.15
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)  | 26.03 | 26.68 | 36.11
| [SeaLLM-7b-Hybrid](https://huggingface.co/SeaLLMs/SeaLLM-7B-Hybrid)  | 27.18 | 26.95 | 36.50
| [SeaLLM-7b-v2](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2)  | 28.48 | 29.92 | 39.18
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)  | 25.75 | 26.15 | 36.28
| [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) | 27.83 | 30.73 | 34.49
| [Sailor-7B](https://huggingface.co/sail/Sailor-7B) | **30.00** | **32.88** | **44.10** | 




| BELEBELE / 3-shot (EM) | Th | Id | Vi 
|-----------| -------  | ------- | ------- | 
| Random    |  <span style="color: gray">25.00</span> | <span style="color: gray">25.00</span> | <span style="color: gray">25.00</span>
| [Qwen1.5-0.5B](https://huggingface.co/Qwen/Qwen1.5-0.5B) | 29.89 | 26.89 | 30.22 
| [Sailor-0.5B](https://huggingface.co/sail/Sailor-0.5B)   | **32.22** | **30.89** | **32.33** |
|-----------| -------  | ------- | ------- |
| [Qwen1.5-1.8B](https://huggingface.co/Qwen/Qwen1.5-1.8B) | 30.11 | 32.00 | 31.33
| [Sailor-1.8B](https://huggingface.co/sail/Sailor-1.8B)   | **34.22** | **34.89** | **35.33** |
|-----------| -------  | ------- | ------- |
| [Qwen1.5-4B](https://huggingface.co/Qwen/Qwen1.5-4B)   | 32.78 | 36.22 | 35.22
| [Sailor-4B](https://huggingface.co/sail/Sailor-4B)   | **36.11** | **41.33** | **38.89** |
|-----------| -------  | ------- | ------- |
| [Llama-2-7b](https://huggingface.co/meta-llama/Llama-2-7b-hf)  | 31.78 | 39.78 | 38.00 |
| [Mistral-7B-v0.1](https://huggingface.co/mistralai/Mistral-7B-v0.1)  | 34.33 | 41.33 | 41.33 |
| [SeaLLM-7b-Hybrid](https://huggingface.co/SeaLLMs/SeaLLM-7B-Hybrid)  | 37.78 | 43.11 | 43.00 |
| [SeaLLM-7b-v2](https://huggingface.co/SeaLLMs/SeaLLM-7B-v2)  | 36.33 | 43.11 | **47.00** |
| [Qwen1.5-7B](https://huggingface.co/Qwen/Qwen1.5-7B)  | 38.33 | 42.00 | 42.89 |
| [Qwen1.5-7B-Chat](https://huggingface.co/Qwen/Qwen1.5-7B-Chat) | 31.22 | 38.22 | 38.67 |
| [Sailor-7B](https://huggingface.co/sail/Sailor-7B) | **41.56** | **44.33** | 45.33 |