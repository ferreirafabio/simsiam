#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -J stn_view_reconstruction_loss
#SBATCH -t 23:59:59 #4-23:59:59 #23:59:59
#SBATCH --array 0-30%1

source /home/ferreira/.profile
conda activate metassl3

random_port=$(shuf -i 10003-15000 -n 1)

#python run_all.py --dist-url "tcp://localhost:$random_port" --multiprocessing-distributed --world-size 1 --rank 0 --expt-name $EXP_NAME --stn_mode $MODE --invert_stn_gradients True --use_stn_optimizer False --pipeline_mode "pretrain" "eval" --epochs 800 --batch-size 2048 --invert_penalty True --stn_conv1_depth 16 --stn_conv2_depth 16 --dim 2048 --pred-dim 512 --penalty_loss ThetaCropsPenalty --summary_writer_freq 10 --view_reconstruction_loss True --lr 0.001

#python run_all.py --dist-url "tcp://localhost:$random_port" --multiprocessing-distributed --world-size 1 --rank 0 --expt-name $EXP_NAME --stn_mode $MODE --invert_stn_gradients True --use_stn_optimizer False --pipeline_mode "pretrain" "eval" --epochs 2000 --batch-size 2048 --invert_penalty True --stn_conv1_depth 16 --stn_conv2_depth 16 --dim 2048 --pred-dim 512 --penalty_loss ThetaCropsPenalty --summary_writer_freq 10 --view_reconstruction_loss2 True --lr 0.001

python run_all.py --dist-url "tcp://localhost:$random_port" --multiprocessing-distributed --world-size 1 --rank 0 --expt-name $EXP_NAME --stn_mode $MODE --invert_stn_gradients True --use_stn_optimizer False --pipeline_mode "pretrain" "eval" --epochs 2000 --batch-size 2048 --invert_penalty True --stn_conv1_depth 16 --stn_conv2_depth 16 --dim 2048 --pred-dim 512 --penalty_loss ThetaCropsPenalty --summary_writer_freq 10 --view_reconstruction_loss2 True --lr 0.001 --epsilon 0.1
