# USAD
echo "run USAD @gpu $1"
sh scripts/USAD.sh $1 USAD toyUSW 12 30 4 1 &&
sh scripts/USAD.sh $1 USAD NeurIPS-TS-UNI 12 30 4 1 &&
sh scripts/USAD.sh $1 USAD NeurIPS-TS-MUL 12 30 12 1 &&
sh scripts/USAD.sh $1 USAD SWaT 12 70  40  5 &&
sh scripts/USAD.sh $1 USAD SMD  5  250 38  5 &&
sh scripts/USAD.sh $1 USAD SMAP 5  250 55  5 &&
sh scripts/USAD.sh $1 USAD MSL  5  250 33  5 &&
sh scripts/USAD.sh $1 USAD WADI 10 70  100 5 &&
echo "complete"
