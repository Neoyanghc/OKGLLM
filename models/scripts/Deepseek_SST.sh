model_name=Deepseek
llm_model=Deepseek
llm_dim=3584
train_epochs=10
learning_rate=0.01
llama_layers=6

master_port=00098
num_process=2
batch_size=64
d_model=16
d_ff=16

export NCCL_P2P_DISABLE=1
export NCCL_IB_DISABLE=1

comment='Deepseek-SST'

CUDA_VISIBLE_DEVICES=0,1 accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /remote-home/share/dmb_nas2/yhc/2025/OceanKG/datasets/OISST/ \
  --data_path sst_noland_5x5_weekly_30.csv\
  --model_id sst_32_32 \
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
