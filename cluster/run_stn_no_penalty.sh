#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -J stn
#SBATCH -t 23:59:59 #4-23:59:59 #23:59:59
#SBATCH --array 0-30%1

source /home/ferreira/.profile
conda activate metassl2

python run_all.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --expt-name $EXP_NAME --stn_mode $MODE --invert_stn_gradients True --use_stn_optimizer False
