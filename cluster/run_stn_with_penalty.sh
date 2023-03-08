#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -J stn_w_pen
#SBATCH -t 23:59:59 #4-23:59:59 #23:59:59
#SBATCH --array 0-30%1

source /home/ferreira/.profile
conda activate metassl2

random_port=$(shuf -i 10003-15000 -n 1)

python run_all.py --dist-url "tcp://localhost:$random_port" --multiprocessing-distributed --world-size 1 --rank 0 --expt-name $EXP_NAME --penalty_loss $PENALTY --epsilon $EPS --penalty_weight $PW --stn_mode $MODE --invert_stn_gradients True --use_stn_optimizer False --pipeline_mode "pretrain" "eval" --epochs 300 --batch-size 2048 --use_unbounded_stn True --invert_penalty True --lr $LR --stn_conv1_depth 8 --stn_conv2_depth 8
