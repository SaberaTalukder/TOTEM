# you have to process each dataset before you can run this file

python imputation/combine_datasets.py

for seed in 2021 13 1
do
for mask_ratio in 0.5 # we only train 1 model at 0.5 masking for all imputation percentages
do
python imputation/train_vqvae.py \
  --config_path imputation/scripts/all.json \
  --model_init_num_gpus 0 \
  --data_init_cpu_or_gpu cpu \
  --comet_log \
  --comet_tag pipeline \
  --comet_name vqvae_all \
  --save_path "imputation/saved_models/all/mask_ratio_"$mask_ratio"/"\
  --base_path "imputation/data"\
  --batchsize 8192 \
  --mask_ratio $mask_ratio \
  --revined_data 'False' \
  --seed $seed
done
done

# uncomment whichever dataset you want to evaluate on
#dataset=electricity
#dataset=ETTh1
#dataset=ETTh2
#dataset=ETTm1
#dataset=ETTm2
dataset=weather

for seed in 2021
do
for mask_ratio_test in 0.125 0.25 0.375 0.5
do
python imputation/imputation_performance.py \
  --dataset $dataset \
  --trained_vqvae_model_path "imputation/saved_models/all/ <fill in right path> /checkpoints/final_model.pth" \
  --compression_factor 4 \
  --gpu 0 \
  --base_path "imputation/data" \
  --mask_ratio $mask_ratio_test
done
done


# for the zero shot datasets you have to process the datasets first (see process_zero_show_data folder)
# uncomment whichever dataset you want to evaluate on
#dataset=neuro2
#dataset=neuro5
#dataset=saugeen
#dataset=sunspot
dataset=us_births

for seed in 2021
do
for mask_ratio_test in 0.125 0.25 0.375 0.5
do
python imputation/imputation_performance.py \
  --dataset $dataset \
  --trained_vqvae_model_path "imputation/saved_models/all/ <fill in right path> /checkpoints/final_model.pth" \
  --compression_factor 4 \
  --gpu 0 \
  --base_path "process_zero_shot_data/data/imputation" \
  --mask_ratio $mask_ratio_test
done
done