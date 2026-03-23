# OKG-LLM
OKG-LLM: Aligning Ocean Knowledge Graph with Numerical Time-Series Data via LLMs for Global Sea Surface Temperature Prediction
https://arxiv.org/abs/2508.00933



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
```

## Quick Demos
1. Tune the model. We provide five experiment scripts for demonstration purposes under the folder `./scripts`.

```
bash ./scripts/OKGLLM_SST.sh 
```

##  Citation

If you find this repo helpful, please cite our paper. 

```
@article{yang2026okg,
  title={OKG-LLM: aligning ocean knowledge graph with observation data via LLMs for global sea surface temperature prediction},
  author={Yang, Hanchen and Wang, Jiaqi and Cao, Jiannong and Li, Wengen and Zheng, Jialun and Li, Yangning and Miao, Chunyu and Guan, Jihong and Zhou, Shuigeng and Yu, Philip S},
  journal={IEEE Transactions on Knowledge and Data Engineering},
  year={2026},
  publisher={IEEE}
}
