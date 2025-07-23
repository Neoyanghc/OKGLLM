# Description: Run DLinear model on SST dataset / no useful RuntimeError: Found dtype Float but expected BFloat16 
model_name=DLinear
train_epochs=10
learning_rate=0.01

master_port=00099
num_process=2
batch_size=1

comment='DLiner-SST'

accelerate launch --multi_gpu --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path /remote-home/share/dmb_nas2/yhc/2025/OceanKG/datasets/OISST/ \
  --data_path sst_5x5_weekly_30.csv\
  --model_id sst_32_32 \
  --model $model_name \
  --data SST \
  --features M \
  --seq_len 32 \
  --label_len 16 \
  --pred_len 32 \
  --e_layers 2 \
  --d_layers 1 \
  --factor 3 \
  --enc_in 2592 \
  --dec_in 2592 \
  --c_out 2592 \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --model_comment $comment
