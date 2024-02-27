python process_zero_shot_data/process_saugeen_sun_births.py \
    --base_path '/path/to/data' \
    --save_path 'process_zero_shot_data/data'

gpu=2
seq_len=96
root_path_name=process_zero_shot_data/data/saugeen/Tin96_Tout96
data_path_name=saugeen
data_name=saugeen
random_seed=2021
pred_len=0

python -u process_zero_shot_data/save_notrevin_notrevinmasked_revinx_revinxmasked.py\
  --random_seed $random_seed \
  --data $data_name \
  --root_path $root_path_name \
  --data_path $data_path_name \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --label_len 0 \
  --enc_in 1\
  --gpu $gpu\
  --save_path "process_zero_shot_data/data/imputation/saugeen"

# draws from imputation processed data
python process_zero_shot_data/prep_data_for_anomaly_detection.py \
    --base_path 'process_zero_shot_data/data/imputation/saugeen/' \
    --save_path 'process_zero_shot_data/data/anomaly_detection/saugeen/'

python process_zero_shot_data/forecasting_saugeen_sun_births.py \
    --base_path '/path/to/data' \
    --save_path 'process_zero_shot_data/data/forecasting'