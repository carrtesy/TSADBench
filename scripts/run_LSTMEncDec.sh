# LSTMEncDec
echo "run LSTMEncDec"
sh scripts/LSTMEncDec.sh 0 LSTMEncDec toyUSW 96 10 128  3 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec NeurIPS-TS-UNI 96 10 128  3 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec NeurIPS-TS-MUL 96 10 128  3 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec SWaT 96 10 128  3  0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec SMD  96 10 38  5 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec SMAP 96 10 55  5 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec MSL  96 10 33  5 0.1 &&
sh scripts/LSTMEncDec.sh 0 LSTMEncDec WADI 96 10 100 5 0.1 &&
echo "complete"
