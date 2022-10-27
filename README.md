# TSAD

This repository contains time series anomaly detection datasets, models, and their implementations.


## Dataset preparation

For list of dataset details, please refer to our [notion dataset page](https://decorous-background-e3e.notion.site/79cb1d595ec746a3a4c8371cedb2c608?v=440fdfeea2dc489d806e72b85d3d4da6). 

All datasets are assumed to be in "data" folder. 

1. Toy Dataset (toyUSW) : We have created toy dataset to test algorithms promptly. [train.npy](data/toyUSW/train.npy) contains periodic sine waves. [test.npy](data/toyUSW/test.npy) has abnormal situations (stopped signal) and anomalies are labeled in file [test_label.npy](data/toyUSW/test_label.npy).  

NeurIPS-TS dataset are created using the principles in https://openreview.net/forum?id=r8IvOsnHchr.
We prepared UNI/MUL dataset, for each data length being 1000.
2. NeurIPS-TS-UNI
3. NeurIPS-TS-MUL

SWaT and WADI dataset has two types of data: train (normal) and test (abnormal).
Train set does not contain anomaly set. Test set has anomalies driven by researcher's attack scenarios.
Request via guidelines in the [link](https://itrust.sutd.edu.sg/itrust-labs_datasets/dataset_info/).

2. SWaT (2022-10-25) : Secure Water Treatment Dataset
- With shared google drive link after request, refer to *SWaT.A1 & A2_Dec 2015*
- For Normal Dataset, refer to ./Physical/SWaT_Dataset_Normal_v0.xlsx
- For Attack Dataset, refer to ./Physical/SWaT_Dataset_Attack_v0.xlsx
- convert xlsx using *read_xlsx_and_convert_to_csv* in utils/tools.py

3. WADI (2022-10-25) : Water Distribution Dataset
- With shared google drive link after request, refer to *WADI.A2_19 Nov 2019*
- For Normal Dataset, refer to ./WADI_14days_new.csv
- For Attack Dataset, refer to ./WADI_attackdataLABLE.csv

SMD, PSM, SMAP, MSL are provided in https://github.com/thuml/Anomaly-Transformer.

4. SMD : Server Machine Dataset

5. PSM : Pooled Server Metrics Dataset

6. SMAP : Soil Moisture Active Passive satellite Dataset

7. MSL : Mars Science Laboratory Dataset

8. To be updated

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
