
<h1 align="center">Leveraging Direct Preference Optimization for LLM Fine-tuning in Knowledge Graph Completion</h1>


## Requirements

Ensure you have Python 3.8 installed along with the following packages:

- **trl**: >=0.8.6
- **Transformers**: >=4.45.2
- **Bitsandbytes**: ==0.43.1
- **Torch**: 2.1.0

Alternatively, you can directly install the environment using:

```bash
pip install -r requirements.txt

```
Please download Meta-Llama-3-8B-Instruct and Gemma-7B from HuggingFace. Make sure to modify the model path before using it. We use a single Nvidia GeForce RTX 4090 GPU with 24GB for our experiments.


## Training model:
commands to run the model in the WN18RR:
```bash
python modelwn18rr.py
```
You can change the path to train on other datasets.
