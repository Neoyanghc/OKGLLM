model_name=OKGLLM
llm_model=BERT
llm_dim=768
train_epochs=10
learning_rate=0.001
llama_layers=6

master_port=00100
num_process=1
batch_size=512
d_model=32
d_ff=64

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

comment='OKG-GPT-SSS'

CUDA_VISIBLE_DEVICES=4 accelerate launch  --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /remote-home/share/dmb_nas2/yhc/2025/OceanKG/datasets/OISSS/ \
  --data_path sss_without_land_data.csv\
  --model_id sss_32_32 \
  --model $model_name \
  --llm_model $llm_model \
  --llm_dim $llm_dim \
  --data SST \
  --features M \
  --seq_len 96 \
  --label_len 16 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 1716 \
  --dec_in 1716 \
  --c_out 1716 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --llm_layers $llama_layers \
  --train_epochs $train_epochs \
  --model_comment $comment
