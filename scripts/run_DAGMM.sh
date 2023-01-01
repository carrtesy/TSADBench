# DAGMM
echo "run DAGMM @gpu $1"
sh scripts/DAGMM.sh $1 DAGMM toyUSW &&
sh scripts/DAGMM.sh $1 DAGMM NeurIPS-TS-UNI &&
sh scripts/DAGMM.sh $1 DAGMM NeurIPS-TS-MUL &&
sh scripts/DAGMM.sh $1 DAGMM SWaT &&
sh scripts/DAGMM.sh $1 DAGMM SMD  &&
sh scripts/DAGMM.sh $1 DAGMM SMAP &&
sh scripts/DAGMM.sh $1 DAGMM MSL  &&
sh scripts/DAGMM.sh $1 DAGMM WADI &&
echo "complete"