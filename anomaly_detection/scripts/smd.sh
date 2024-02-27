python anomaly_detection/save_chunked_data.py \
    --data 'SMD' \
    --batch_size 128 \
    --task_name 'anomaly_detection' \
    --root_path "/path/to/raw/data" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/SMD/" \
    --num_vars 38

python anomaly_detection/revin_data.py \
    --root_path "anomaly_detection/data/SMD/" \
    --seq_len 100 \
    --save_path "anomaly_detection/data/SMD/revin_data/" \
    --num_vars 38

for seed in 47 1 13
do
python anomaly_detection/train_vqvae.py \
    --config_path anomaly_detection/scripts/smd.json \
    --model_init_num_gpus 0 \
    --data_init_cpu_or_gpu cpu \
    --comet_log \
    --comet_tag pipeline \
    --comet_name smd \
    --save_path "anomaly_detection/saved_models/SMD/" \
    --base_path "anomaly_detection/data/SMD/revin_data/"\
    --batchsize 4096 \
    --seed $seed
done

seed=47
python anomaly_detection/detect_anomaly.py \
    --dataset "SMD"\
    --trained_vqvae_model_path "anomaly_detection/saved_models/ <fill in  proper path> _seed"$seed"/checkpoints/final_model.pth" \
    --compression_factor 4 \
    --base_path "anomaly_detection/data/SMD/revin_data"\
    --labels_path "anomaly_detection/data/SMD"\
    --anomaly_ratio 0.5 \
    --gpu 0 \
    --num_vars 38 \
    --seq_len 100