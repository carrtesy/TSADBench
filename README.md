# TSAD

This repository contains time series anomaly detection datasets, models, and their implementations.


## Dataset preparation
We don't have rights to publicly distribute dataset we get.

Hence, we provide ways to get data instead. 

All datasets are assumed to be in "data" folder. 

1. Toy Dataset: We have created toy dataset to test algorithms promptly. [train_X.csv](data/toy/train_X.csv) contains periodic sine waves. [test_X.csv](data/toy/train_X.csv) has abnormal situations (zero signal) and anomalies are labeled in file [test_y.csv](data/toy/test_y.csv).  
2. SWaT (https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/): SWaT and WADI dataset has two types of data: train (normal) and test (abnormal).
Train set does not contain anomaly set. Test set has anomalies driven by researcher's attack scenarios.
3. *TO BE UPDATED*


## Anomaly detection models

1. LSTM Enc-Dec structure
[Malhotra, Pankaj, et al. "LSTM-based encoder-decoder for multi-sensor anomaly detection."(2016).](https://arxiv.org/pdf/1607.00148v2.pdf)
2. OmniAnomaly
[Su, Ya, et al. "Robust anomaly detection for multivariate time series through stochastic recurrent neural network." Proceedings of the 25th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2019.](https://dl.acm.org/doi/pdf/10.1145/3292500.3330672?casa_token=k52TYpPsw2QAAAAA:5PQRaCv7bH507y-pnpvFqLM_TDUmMMTlZU24P8coKzZmT6LVtFC-8dh8AmhTJ_kYZFl11NyxBSGi)
3. USAD
[Audibert, Julien, et al. "Usad: Unsupervised anomaly detection on multivariate time series." Proceedings of the 26th ACM SIGKDD International Conference on Knowledge Discovery & Data Mining. 2020.](https://dl.acm.org/doi/pdf/10.1145/3394486.3403392)

## Paper implementation
- SWaT

| Model       | Accuracy | Precision | Recall | F1 |
|-------------|----------|-----------|--------|----|
| LSTMAE      |          |           |        |    |
| OmniAnomaly |          |           |        |    |
| USAD        |          |           |        |    |