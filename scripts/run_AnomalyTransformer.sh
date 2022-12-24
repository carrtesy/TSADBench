# AnomalyTransformer
echo "run AnomalyTransformer @gpu $1"
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer toyUSW 100 3 0.5 3&&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer NeurIPS-TS-UNI 100 3 0.5 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer NeurIPS-TS-MUL 100 3 0.5 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer SWaT 100 3 0.5 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer SMD  100 10 0.5 3  &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer SMAP 100 3 1 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer MSL  100 3 1 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer WADI 100 3 1 3 &&
sh scripts/AnomalyTransformer.sh $1 AnomalyTransformer PSM  100 3 1 3 &&
echo "complete"