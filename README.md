# OKG-LLM
OKG-LLM: Aligning Ocean Knowledge Graph with Numerical Time-Series Data via LLMs for Global Sea Surface Temperature Prediction


## Requirements
Use python 3.11 from MiniConda

- torch==2.2.2
- accelerate==0.28.0
- einops==0.7.0
- matplotlib==3.7.0
- numpy==1.23.5
- pandas==1.5.3
- scikit_learn==1.2.2
- scipy==1.12.0
- tqdm==4.65.0
- peft==0.4.0
- transformers==4.31.0
- deepspeed==0.14.0
- sentencepiece==0.2.0

To install all dependencies:
```
pip install -r requirements.txt


## Quick Demos
1. Tune the model. We provide five experiment scripts for demonstration purposes under the folder `./scripts`.

```bash
bash ./scripts/OKGLLM_SST.sh 
