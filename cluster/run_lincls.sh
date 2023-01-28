#!/bin/zsh
#SBATCH -p alldlc_gpu-rtx2080 #bosch_gpu-rtx2080 #alldlc_gpu-rtx2080
#SBATCH --gres=gpu:4
#SBATCH -J simsiam_lincls
#SBATCH -t 23:59:59 #4-23:59:59 #23:59:59

source /home/ferreira/.profile
source activate metassl

python main_lincls.py --dist-url 'tcp://localhost:10001' --multiprocessing-distributed --world-size 1 --rank 0 --pretrained "experiments/cifar10/checkpoint_0799.pth.tar" --dataset "CIFAR10" --expt-name "cifar10"

