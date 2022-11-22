# USAD
echo "run USAD"
sh scripts/run_USAD.sh 0 USAD toyUSW 12 30 4 1 > ./log/USAD_toyUSW &&
sh scripts/run_USAD.sh 0 USAD NeurIPS-TS-UNI 12 30 4 1 > ./log/USAD_NeurIPS-TS-UNI &&
sh scripts/run_USAD.sh 0 USAD NeurIPS-TS-MUL 12 30 12 1 > ./log/USAD_NeurIPS-TS-MUL &&
#sh scripts/run_USAD.sh 0 USAD SWaT 12 70  40  5 > ./log/USAD_SWaT &&
#sh scripts/run_USAD.sh 0 USAD SMD  5  250 38  5 > ./log/USAD_SMD  &&
#sh scripts/run_USAD.sh 0 USAD SMAP 5  250 55  5 > ./log/USAD_SMAP &&
#sh scripts/run_USAD.sh 0 USAD MSL  5  250 33  5 > ./log/USAD_MSL  &&
#sh scripts/run_USAD.sh 0 USAD WADI 10 70  100 5 > ./log/USAD_WADI &&
echo "complete"