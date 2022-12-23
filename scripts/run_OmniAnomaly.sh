# OmniAnomaly
echo "run OmniAnomaly @gpu $1"
sh scripts/OmniAnomaly.sh $1 OmniAnomaly SWaT 96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly SMD  96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly SMAP 96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly MSL  96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly WADI 96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly toyUSW 96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly NeurIPS-TS-UNI 96 10 128 128 128 &&
sh scripts/OmniAnomaly.sh $1 OmniAnomaly NeurIPS-TS-MUL 96 10 128 128 128 &&
echo "complete"