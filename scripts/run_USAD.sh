# USAD
echo "run USAD"
sh scripts/USAD.sh 0 USAD toyUSW 12 30 4 1 &&
sh scripts/USAD.sh 0 USAD NeurIPS-TS-UNI 12 30 4 1 &&
sh scripts/USAD.sh 0 USAD NeurIPS-TS-MUL 12 30 12 1 &&
sh scripts/USAD.sh 0 USAD SWaT 12 70  40  5 &&
sh scripts/USAD.sh 0 USAD SMD  5  250 38  5 &&
sh scripts/USAD.sh 0 USAD SMAP 5  250 55  5 &&
sh scripts/USAD.sh 0 USAD MSL  5  250 33  5 &&
sh scripts/USAD.sh 0 USAD WADI 10 70  100 5 &&
echo "complete"
