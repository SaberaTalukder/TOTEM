# have to process all datasets independently before running this

python forecasting/combine_datasets_for_vqvae.py \
    --root_path forecasting/data \
    --save_path forecasting/data/all/

gpu=0
python forecasting/train_vqvae.py \
  --config_path forecasting/scripts/all.json \
  --model_init_num_gpus $gpu \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_all\
  --save_path "forecasting/saved_models/all/"\
  --base_path "forecasting/data"\
  --batchsize 4096

# --------------------------------------------------------------
# extract the weather, electricity, traffic, ETT datasets using the all vqvae

random_seed=2021
root_path_name=/path/to/original/files
data_path_name=electricity.csv
model_id_name=electricity
data_name=custom
seq_len=96
gpu=0
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
  --enc_in 321 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/electricity/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

seq_len=96
random_seed=2021
root_path_name=/path/to/original/files
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
  --save_path "forecasting/data/all_vqvae_extracted/ETTh1/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

seq_len=96
random_seed=2021
root_path_name=/path/to/original/files
data_path_name=ETTh2.csv
model_id_name=ETTh2
data_name=ETTh2
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
  --save_path "forecasting/data/all_vqvae_extracted/ETTh2/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=/path/to/original/files
data_path_name=ETTm1.csv
model_id_name=ETTm1
data_name=ETTm1
seq_len=96
gpu=0
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
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/ETTm1/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=/path/to/original/files
data_path_name=ETTm2.csv
model_id_name=ETTm2
data_name=ETTm2
seq_len=96
gpu=0
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
  --enc_in 7 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/ETTm2/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

random_seed=2021
root_path_name=/path/to/original/files
data_path_name=traffic.csv
model_id_name=traffic
data_name=custom
seq_len=96
gpu=0
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
  --enc_in 862 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/traffic/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
root_path_name=/path/to/original/files
data_path_name=weather.csv
model_id_name=weather
data_name=custom
seq_len=96
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
  --enc_in 21 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/weather/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done


# --------------------------------------------------------------

python forecasting/combine_datasets_for_forecaster.py \
    --root_path forecasting/data/all_vqvae_extracted \
    --save_path forecasting/data/all_vqvae_extracted/

gpu=0
Tin=96
datatype=all
for seed in 2021 1 13
do
for Tout in 96 # 192 336 720 # uncomment these when want to run all forecasting lengths
do
python forecasting/train_forecaster.py \
  --data-type $datatype \
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu \
  --seed $seed \
  --data_path "forecasting/data/"$datatype"_vqvae_extracted/Tin"$Tin"_Tout"$Tout"" \
  --codebook_size 256 \
  --checkpoint \
  --checkpoint_path "forecasting/saved_models/"$datatype"/forecaster_checkpoints/"$datatype"_Tin"$Tin"_Tout"$Tout"_seed"$seed""\
  --file_save_path "forecasting/results/"$datatype"/"
done
done

gpu=1
Tin=96
Tout=96 # 192 336 720 # uncomment these when want to run all forecasting lengths
seed=2021
for dt in electricity ETTh1 ETTh2 ETTm1 ETTm2 traffic weather
do
python forecasting/generalist_eval.py \
  --data-type $dt \
  --data_path "forecasting/data/all_vqvae_extracted/"$dt"/Tin"$Tin"_Tout"$Tout"/" \
  --model_load_path "forecasting/saved_models/all/forecaster_checkpoints/all_Tin"$Tin"_Tout"$Tout"_seed"$seed"/"\
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu
done

# --------------------------------------------------------------
# zero shot
# you have to run the scripts in the process_zero_shot_data folder before you can run this

gpu=1
random_seed=2021
data_path_name=saugeen_web
model_id_name=saugeen_web
data_name=saugeen_web
seq_len=96
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "process_zero_shot_data/data/forecasting/saugeen/Tin"$seq_len"_Tout"$pred_len"/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \t
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/saugeen/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
data_path_name=saugeen_web  # stays saugeen_web even though data is sunspot
model_id_name=saugeen_web  # stays saugeen_web even though data is sunspot
data_name=saugeen_web  # stays saugeen_web even though data is sunspot
seq_len=96
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "process_zero_shot_data/data/forecasting/sunspot/Tin"$seq_len"_Tout"$pred_len"/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/sunspot/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
data_path_name=saugeen_web  # stays saugeen_web even though data is us_births
model_id_name=saugeen_web  # stays saugeen_web even though data is us_births
data_name=saugeen_web  # stays saugeen_web even though data is us_births
seq_len=96
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "process_zero_shot_data/data/forecasting/us_births/Tin"$seq_len"_Tout"$pred_len"/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/us_births/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
data_path_name=neuro2
model_id_name=neuro2
data_name=neuro
seq_len=96
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "process_zero_shot_data/data/pt2/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 72 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/neuro2/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
random_seed=2021
data_path_name=neuro5
model_id_name=neuro5
data_name=neuro
seq_len=96
for pred_len in 96 192 336 720
do
python -u forecasting/extract_forecasting_data.py \
  --random_seed $random_seed \
  --data $data_name \
  --root_path "process_zero_shot_data/data/pt5/" \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 106 \
  --gpu $gpu\
  --save_path "forecasting/data/all_vqvae_extracted/neuro5/Tin"$seq_len"_Tout"$pred_len"/"\
  --trained_vqvae_model_path '/path/to/all/trained/vqvae'\
  --compression_factor 4 \
  --classifiy_or_forecast "forecast"
done

gpu=1
Tin=96
Tout=96 # 192 336 720 # uncomment when want all forecasting lengths
seed=2021
for dt in saugeen us_births sunspot neuro2 neuro5
do
python forecasting/generalist_eval.py \
  --data-type $dt \
  --data_path "forecasting/data/all_vqvae_extracted/"$dt"/Tin"$Tin"_Tout"$Tout"/" \
  --model_load_path "forecasting/saved_models/all/forecaster_checkpoints/all_Tin"$Tin"_Tout"$Tout"_seed"$seed"/"\
  --Tin $Tin \
  --Tout $Tout \
  --cuda-id $gpu
done