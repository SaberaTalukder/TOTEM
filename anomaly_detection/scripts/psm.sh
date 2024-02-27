python anomaly_detection/save_chunked_data.py \
    --data 'PSM' \
    --batch_size 128 \
    --task_name 'anomaly_detection' \
    --root_path "/path/to/raw/data" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/PSM/" \
    --num_vars 25

python anomaly_detection/revin_data.py \
    --root_path "anomaly_detection/data/PSM/" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/PSM/revin_data/" \
    --num_vars 25

for seed in 47 1 13
do
python anomaly_detection/train_vqvae.py \
    --config_path anomaly_detection/scripts/psm.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --comet_log \
    --comet_tag pipeline \
    --comet_name psm \
    --save_path "anomaly_detection/saved_models/PSM/" \
    --base_path "anomaly_detection/data/PSM/revin_data/"\
    --batchsize 4096 \
    --seed $seed
done

seed=47
python anomaly_detection/detect_anomaly.py \
    --dataset "PSM"\
    --trained_vqvae_model_path "anomaly_detection/saved_models/ <fill in  proper path> _seed"$seed"/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "anomaly_detection/data/PSM/revin_data"\
    --labels_path "anomaly_detection/data/PSM"\
    --anomaly_ratio 1 \
    --gpu 0 \
    --num_vars 25 \
    --seq_len 100