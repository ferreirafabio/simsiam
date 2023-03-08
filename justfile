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

@stn_with_penalty EXP_NAME MODE PENALTY EPS PW LR:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}},LR={{LR}} cluster/run_stn_with_penalty.sh

@stn_with_penalty_ema_4way EXP_NAME MODE PENALTY EPS PW LR INV_PENALTY:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}},LR={{LR}},INV_PENALTY={{INV_PENALTY}} cluster/run_stn_with_penalty_ema_four_way.sh

@stn_no_penalty_4way EXP_NAME MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}} cluster/run_stn_no_penalty_four_way_loss.sh

@stn_lin_eval EXP_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}} cluster/run_lincls.sh

@wo_stn EXP_NAME:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}} cluster/run_wo_stn.sh

@stn_theta_prediction_loss EXP_NAME MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}} cluster/run_stn_theta_prediction_loss.sh

@stn_view_reconstruction EXP_NAME MODE:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}} cluster/run_stn_view_reconstruction.sh

@stn_with_penalty_ema EXP_NAME MODE PENALTY EPS PW LR INV_PENALTY:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}},LR={{LR}},INV_PENALTY={{INV_PENALTY}} cluster/run_stn_with_penalty_ema.sh


@stn_with_penalty_ema_long EXP_NAME MODE PENALTY EPS PW LR INV_PENALTY:
   #!/usr/bin/env zsh
   mkdir -p /work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}
   sbatch --output=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --error=/work/dlclarge1/ferreira-simsiam/simsiam/experiments/{{EXP_NAME}}/%x.%A.%a.%N.err_out --export=EXP_NAME={{EXP_NAME}},MODE={{MODE}},PENALTY={{PENALTY}},EPS={{EPS}},PW={{PW}},LR={{LR}},INV_PENALTY={{INV_PENALTY}} cluster/run_stn_with_penalty_ema_long_training.sh
