#!/bin/bash
for data in toyUSW NeurIPS-TS-UNI NeurIPS-TS-MUL MSL PSM SMAP SWaT WADI SMD
do
    for model in OCSVM IsolationForest LOF
    do
      echo "running: model $model data $data"
      sh scripts/sklearn.sh $model $data
    done
done