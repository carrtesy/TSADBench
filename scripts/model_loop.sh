#!/bin/bash
echo "dataset $1 gpu $2"
for model in RandomModel OCSVM IsolationForest LOF LSTMEncDec LSTMVAE OmniAnomaly USAD DeepSVDD DAGMM THOC AnomalyTransformer
do
  echo "$model + $1"
  sh scripts/$1/$model.sh $2
done