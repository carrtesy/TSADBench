# DeepSVDD
echo "run DeepSVDD @gpu $1"
sh scripts/DeepSVDD.sh $1 DeepSVDD toyUSW &&
sh scripts/DeepSVDD.sh $1 DeepSVDD NeurIPS-TS-UNI &&
sh scripts/DeepSVDD.sh $1 DeepSVDD NeurIPS-TS-MUL &&
sh scripts/DeepSVDD.sh $1 DeepSVDD SWaT &&
sh scripts/DeepSVDD.sh $1 DeepSVDD SMD  &&
sh scripts/DeepSVDD.sh $1 DeepSVDD SMAP &&
sh scripts/DeepSVDD.sh $1 DeepSVDD MSL  &&
sh scripts/DeepSVDD.sh $1 DeepSVDD WADI &&
echo "complete"