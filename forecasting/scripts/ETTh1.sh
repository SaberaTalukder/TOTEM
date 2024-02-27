seq_len=96
root_path_name=/add/your/own/path
data_path_name=ETTh1.csv
data_name=ETTh1
random_seed=2021
pred_len=96
gpu=0

python -u forecasting/save_revin_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/ETTh1"

gpu=0
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/ETTh1.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_ETTh1 \
  --save_path "forecasting/saved_models/ETTh1/"\
  --base_path "forecasting/data"\
  --batchsize 4096


seq_len=96
random_seed=2021
root_path_name=/add/your/own/path/to/original/files
data_path_name=ETTh1.csv
model_id_name=ETTh1
data_name=ETTh1
gpu=1
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 7\
  --gpu $gpu\
  --save_path "forecasting/data/ETTh1/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path 'add/path/to/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=0
Tin=96
datatype=ETTh1
for seed in 2021 1 13
do
for Tout in 96 192 336 720
do
python forecasting/train_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done