# you have to process each dataset before you can run this file

python anomaly_detection/combine_datasets.py \
    --root_path 'anomaly_detection/data' \
    --save_path 'anomaly_detection/data/ALL/revin_data/' \
    --gpu 0

python anomaly_detection/combine_labels.py \
    --root_path 'anomaly_detection/data' \
    --save_path 'anomaly_detection/data/ALL/' \
    --gpu 0

for seed in 47 1 13
do
python anomaly_detection/train_vqvae.py \
    --config_path anomaly_detection/scripts/all.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --comet_log \
    --comet_tag pipeline \
    --comet_name all \
    --save_path "anomaly_detection/saved_models/ALL/" \
    --base_path "anomaly_detection/data/ALL/revin_data/"\
    --batchsize 4096 \
    --seed $seed
done

# uncomment whichever dataset below you want to evaluate on
#dt=SMD
#num_vars=38
#ar=0.5

#dt=MSL
#num_vars=55
#ar=2

#dt=PSM
#num_vars=25
#ar=1

#dt=SMAP
#num_vars=25
#ar=1

dt=SWAT
num_vars=51
ar=1

seed=47
python anomaly_detection/detect_anomaly.py \
       --dataset $dt\
       --trained_vqvae_model_path "anomaly_detection/saved_models/ALL/ <fill in  proper path> _seed"$seed"/checkpoints/final_model.pth" \
       --compression_factor 4 \
       --base_path "anomaly_detection/data/"$dt"/revin_data"\
       --labels_path "anomaly_detection/data/"$dt""\
       --anomaly_ratio $ar \
       --gpu 0\
       --num_vars $num_vars \
       --seq_len 100

# for the zero shot datasets you have to process the datasets first (see process_zero_show_data folder)
# uncomment whichever dataset below you want to evaluate on
#dt=neuro2
#num_vars=1
#ar=2

#dt=neuro5
#num_vars=1
#ar=2

#dt=saugeen
#num_vars=1
#ar=2

#dt=sunspot
#num_vars=1
#ar=2

dt=us_births
num_vars=1
ar=2

seed=47
python anomaly_detection/detect_anomaly.py \
       --dataset $dt\
       --trained_vqvae_model_path "anomaly_detection/saved_models/ALL/ <fill in  proper path> _seed"$seed"/checkpoints/final_model.pth" \
       --compression_factor 4 \
       --base_path "process_zero_shot_data/data/anomaly_detection/"$dt"/"\
       --labels_path "process_zero_shot_data/data/anomaly_detection/"$dt"/"\
       --anomaly_ratio $ar \
       --gpu 0\
       --num_vars $num_vars \
       --seq_len 96