python process_zero_shot_data/process_neuro_data.py \
    --patient_num 2 \
    --base_path '/path/to/nc/file/' \
    --save_path 'process_zero_shot_data/data/'

gpu=0
seq_len=96
root_path_name=process_zero_shot_data/data/pt2
data_path_name=neuro2
data_name=neuro
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
  --enc_in 72\
  --gpu $gpu\
  --save_path "process_zero_shot_data/data/imputation/neuro2"

# draws from imputation processed data
python process_zero_shot_data/prep_data_for_anomaly_detection.py \
    --base_path 'process_zero_shot_data/data/imputation/neuro2/' \
    --save_path 'process_zero_shot_data/data/anomaly_detection/neuro2/'

# do not need anything further for forecasting