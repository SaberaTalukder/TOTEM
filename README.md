# TOTEM: TOkenized Time series EMbeddings for General Time Series Analysis
TOTEM explores time series unification through discrete tokens (not patches!!). Its simple VQVAE backbone learns a self-supervised, discrete, codebook in either a generalist (multiple domains) or specialist (1 domain) manner.
TOTEM's codebook can then be tested on in domain or zero shot data with many 🔥 time series tasks.

For a high level overview see the [video recap](https://www.youtube.com/watch?v=OqrCpdb6MJk).
Check out the [paper](https://arxiv.org/pdf/2402.16412.pdf) for more details!

## Get Started with TOTEM 💪

### 1. Setup your environment 🤓
```
pip install -r requirements.txt
```

### 2. Get the [data](https://drive.google.com/drive/u/0/folders/1gI36rS8irRZ32ibzKBPGncDmMXQtEf1C) ⏳

### 3. Run TOTEM 🚀

```
# Imputation Specialist
imputation/scripts/electricity.sh or ETTh1.sh or ETTh2.sh or ETTm1.sh or ETTm2.sh or weather.sh

# Imputation Generalist
imputation/scripts/all.sh

# Anomaly Detection Specialist
anomaly_detection/scripts/msl.sh or psm.sh or smap.sh or smd.sh or swat.sh

# Anomaly Detection Generalist
anomaly_detection/scripts/all.sh

# Forecasting Specialist
forecasting/scripts/electricity.sh or ETTh1.sh or ETTh2.sh or ETTm1.sh or ETTm2.sh or weather.sh or traffic.sh

# Forecasting Generalist
forecasting/scripts/all.sh

# Process Zero Shot Data
process_zero_shot_data/scripts/neuro2.sh or neuro5.sh or saugeen.sh or sunspot.sh or us_births.sh
```

### 4. Model Zoo (a.k.a Pretrained Models) 🦑🐯🐊🐳
Find the pretrained generalist tokenizers [here](https://drive.google.com/drive/u/0/folders/1TSwPHDMAhcpe2AKl4xsVbUUmAvd_Tp-Z).
Read some notes on usage [here](https://docs.google.com/document/d/1GbqYFBsTZWKoXu2yFs49sP9X311q8BsnjNsfHYYN_Wg/edit?tab=t.0).


## Cite If You ❤️ TOTEM

```
@article{
talukder2024totem,
title={{TOTEM}: {TO}kenized Time Series {EM}beddings for General Time Series Analysis},
author={Sabera J Talukder and Yisong Yue and Georgia Gkioxari},
journal={Transactions on Machine Learning Research},
issn={2835-8856},
year={2024},
url={https://openreview.net/forum?id=QlTLkH6xRC},
note={}
}
