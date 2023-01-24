#!/bin/bash
echo "model $1 gpu $2"
for dataset in MSL NeurIPS-TS-MUL NeurIPS-TS-UNI PSM SMAP SMD SWaT toyUSW WADI
do
  echo "$1 + $dataset"
  sh scripts/$dataset/$1.sh $2
done