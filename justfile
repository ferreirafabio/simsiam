# List available just commands
@list:
  just --list

@simsiam_lincls EXPERIMENT_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXPERIMENT_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXPERIMENT_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXPERIMENT_NAME}}/%x.%A.%a.%N.err_out --export=EXPERIMENT_NAME={{EXPERIMENT_NAME}} cluster/run_lincls.sh
