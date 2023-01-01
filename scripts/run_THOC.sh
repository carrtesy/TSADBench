# THOC
echo "run THOC @gpu $1"
sh scripts/THOC.sh $1 THOC toyUSW 1e-03 48 1 1 1 128 &&
sh scripts/THOC.sh $1 THOC NeurIPS-TS-UNI 1e-03 48 1 1 1 128 &&
sh scripts/THOC.sh $1 THOC NeurIPS-TS-MUL 1e-03 48 1 1 1 128 &&
sh scripts/THOC.sh $1 THOC SWaT 0.01 18 0.59 1.32 0.34 128 &&
sh scripts/THOC.sh $1 THOC SMD  1e-03 48 1 1 1 128 &&
sh scripts/THOC.sh $1 THOC SMAP 0.09 70 0.5 1.2 1.4 128 &&
sh scripts/THOC.sh $1 THOC MSL  0.09 93 1.73 0.66 1.36 128 &&
sh scripts/THOC.sh $1 THOC WADI 1e-03 48 1 1 1 128&&
echo "complete"