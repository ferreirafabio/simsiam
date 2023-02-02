# List available just commands
@list:
  just --list

@stn EXP_NAME MODE PENALTY EPS PW:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}} cluster/run_stn.sh

@stn_no_penalty EXP_NAME MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}} cluster/run_stn_no_penalty.sh

@stn_no_freeze EXP_NAME MODE PENALTY EPS PW:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}} cluster/run_stn_no_stn_freeze.sh
