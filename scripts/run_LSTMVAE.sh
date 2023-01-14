# LSTMVAE
echo "run LSTMVAE @gpu $1"
sh scripts/LSTMVAE.sh $1 LSTMVAE toyUSW 1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE NeurIPS-TS-UNI 1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE NeurIPS-TS-MUL 1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE SWaT 1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE SMD  1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE SMAP 1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE MSL  1e-03 32 3 2 1e-04 &&
sh scripts/LSTMVAE.sh $1 LSTMVAE WADI 1e-03 32 3 2 1e-04 &&
echo "complete"
