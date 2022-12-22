#!/bin/bash
for data in toyUSW NeurIPS-TS-UNI NeurIPS-TS-MUL SWaT WADI SMD PSM SMAP MSL
do
    for model in OCSVM IsolationForest LOF
    do
      echo "running: model $model data $data"
      sh scripts/sklearn.sh $model $data
    done
done