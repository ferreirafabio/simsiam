from main_simsiam import main as run_simiam
from main_lincls import main as run_linear
import torchvision.models as models
import argparse
import utils
import time
import os
import penalties
import datetime
import sys
import ConfigSpace as CS
from ConfigSpace import InCondition, EqualsCondition, NotEqualsCondition 
import ConfigSpace.hyperparameters as CSH
from copy import deepcopy
from bohb_optim import run_bohb_parallel
from argparse import Namespace
import pathlib

import logging
logging.basicConfig(level=logging.WARNING)


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))

penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}

parser = argparse.ArgumentParser(description='SimSiam Full Pipeline')
parser.add_argument("--dataset_path", type=str, default="/data/datasets/ImageNet/imagenet-pytorch", help="path to dataset")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=300, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 2048), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 0.0005)',
                    dest='weight_decay')
parser.add_argument('--stn-wd', '--stn-weight-decay', default=1e-5, type=float,
                    metavar='W', help='',
                    dest='stn_weight_decay')

parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--world-size', default=-1, type=int,
                    help='number of nodes for distributed training')
parser.add_argument('--rank', default=-1, type=int,
                    help='node rank for distributed training')
parser.add_argument('--dist-url', default='tcp://224.66.41.62:23456', type=str,
                    help='url used to set up distributed training')
parser.add_argument('--dist-backend', default='nccl', type=str,
                    help='distributed backend')
parser.add_argument('--seed', default=None, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=None, type=int,
                    help='GPU id to use.')
parser.add_argument('--multiprocessing-distributed', action='store_true',
                    help='Use multi-processing distributed training to launch '
                         'N processes per node, which has N GPUs. This is the '
                         'fastest way to use PyTorch for either single node or '
                         'multi node data parallel training')

parser.add_argument('--dataset', default='CIFAR10', type=str, help='dataset name. Should be one of ImageNet or CIFAR10.')
parser.add_argument('--expt-name', default='default_experiment', type=str, help='Name of the experiment')
parser.add_argument('--save-freq', default=100, type=int, metavar='N', help='checkpoint save frequency (default: 10)')

# STN
parser.add_argument("--invert_stn_gradients", default=True, type=utils.bool_flag,
                    help="Set this flag to invert the gradients used to learn the STN")
parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate optimizer for the STN parameters; "
                            "annealed with cosine and no warmup")
parser.add_argument('--stn_mode', default='translation_scale_symmetric', type=str,
                    help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                            'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
parser.add_argument("--stn_lr", default=5e-5, type=float, help="""Learning rate at the end of
                    linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
                    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate localization network for each head.")
parser.add_argument("--summary_writer_freq", default=50, type=int, 
help="Defines the number of iterations the summary writer will write output.")
parser.add_argument("--grad_check_freq", default=100, type=int,
                    help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
parser.add_argument("--summary_plot_size", default=16, type=int,
                    help="Defines the number of samples to show in the summary writer.")
parser.add_argument('--stn_pretrained_weights', default='', type=str,
                    help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
parser.add_argument("--deep_loc_net", default=False, type=utils.bool_flag,
                    help="(legacy) Set this flag to use a deep loc net. (default: False).")
parser.add_argument("--stn_res", default=(32, 32), type=int, nargs='+',
                    help="Set the resolution of the global and local crops of the STN (default: 32x and 32x)")
parser.add_argument("--use_unbounded_stn", default=True, type=utils.bool_flag,
                    help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                    help="Specifies the number of warmup epochs for the STN (default: 0).")
parser.add_argument("--stn_conv1_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
parser.add_argument("--stn_conv2_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv2 for the STN localization network (default: 32).")
parser.add_argument("--stn_theta_norm", default=True, type=utils.bool_flag,
                    help="Set this flag to normalize 'theta' in the STN before passing to affine_grid(theta, ...). Fixes the problem with cropping of the images (black regions)")
parser.add_argument("--penalty_loss", default="", type=str, choices=penalty_list,
                    help="Specify the name of the similarity to use.")
parser.add_argument("--epsilon", default=1., type=float,
                    help="Scalar for the penalty loss")
parser.add_argument("--invert_penalty", default=True, type=utils.bool_flag,
                    help="Invert the penalty loss.")
parser.add_argument("--stn_color_augment", default=False, type=utils.bool_flag, help="todo")
parser.add_argument("--resize_all_inputs", default=True, type=utils.bool_flag,
                    help="Resizes all images of the ImageNet dataset to one size. Here: 224x224")
parser.add_argument("--resize_input", default=False, type=utils.bool_flag,
                    help="Set this flag to resize the images of the dataset, images will be resized to the value given "
                            "in parameter --resize_size (default: 512). Can be useful for datasets with varying resolutions.")
parser.add_argument("--resize_size", default=512, type=int,
                    help="If resize_input is True, this will be the maximum for the longer edge of the resized image.")

parser.add_argument('--global_crops_scale', type=float, nargs='+', default=(0.4, 1.),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
                    Used for large global view cropping. When disabling multi-crop (--local_crops_number 0), we
                    recommand using a wider range of scale ("--global_crops_scale 0.14 1." for example)""")
parser.add_argument('--local_crops_number', type=int, default=0, 
                    help="""Number of small local views to generate. Set this parameter to 0 to disable multi-crop training. 
                    When disabling multi-crop we recommend to use "--global_crops_scale 0.14 1." """)
parser.add_argument('--local_crops_scale', type=float, nargs='+', default=(0.05, 0.4),
                    help="""Scale range of the cropped image before resizing, relatively to the origin image.
                    Used for small local view cropping of multi-crop.""")
parser.add_argument("--warmstart_backbone", default=False, type=utils.bool_flag, help="used to load an already trained backbone and set start_epoch to 0.")
parser.add_argument("--penalty_weight", default=1, type=int, help="Specifies the weight for the penalty term.")
parser.add_argument("--stn_ema_update", default=False, type=utils.bool_flag, help="")
parser.add_argument("--stn_ema_momentum", default=0.998, type=int, help="")
parser.add_argument("--penalty_target", default='mean', type=str, choices=['zero', 'one', 'mean', 'rand'],
                        help="Specify the type of target of the penalty. Here, the target is the area with respect to"
                             "the original image. `zero` and `one` are the values itself. `mean` and `rand` are"
                             "inferred with respect to given crop-scales.")
parser.add_argument("--min_glb_overlap", default=0.6, type=float, help="The minimal overlap between the two global crops.")
parser.add_argument("--min_lcl_overlap", default=0.1, type=float, help="The minimal overlap between two local crops.")

parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use_pretrained_stn', default=False, type=utils.bool_flag, metavar='PATH',
                    help='')

parser.add_argument('--four_way_loss', default=False, type=utils.bool_flag, help='')
parser.add_argument('--theta_prediction_loss', default=False, type=utils.bool_flag, help='')
parser.add_argument('--use_stn', default=True, type=utils.bool_flag, help='for testing reasons')


# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help="""Path to pretrained weights of the STN network. 
                    "If specified, the STN is not trained and used to 
                    "pre-process images solely.""")

parser.add_argument("--pipeline_mode", default=('pretrain', 'frozen', 'eval'), type=str, nargs='+', help="")

# BOHB
parser.add_argument('--n_workers', type=int, help='Number of workers to run in parallel.', default=1)
parser.add_argument('--is_worker', help='Flag to turn this into a worker process', type=utils.bool_flag, default=True)
parser.add_argument('--run_id', type=str, help='A unique run id for this optimization run. An easy option is to use the job id of the clusters scheduler.')


class ExperimentWrapper():
    def get_bohb_parameters(self):
        params = {}
        params['min_budget'] = 0.01
        params['max_budget'] = 1
        params['eta'] = 3
        params['iterations'] = 1000
        params['random_fraction'] = 0.333

        return params

    def get_configspace(self):
        cs = CS.ConfigurationSpace()
                
        # stn_wd = CSH.UniformFloatHyperparameter(name='stn_weight_decay', lower=1e-6, upper=0.005, log=True, default_value=0.0005)
        # cs.add_hyperparameter(stn_wd)

        # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='invert_stn_gradients', choices=[True, False], default_value=True))
        
        stn_optimizer = CSH.CategoricalHyperparameter(name='use_stn_optimizer', choices=[True, False], default_value=False)
        cs.add_hyperparameter(stn_optimizer)

        stn_lr = CSH.UniformFloatHyperparameter(name='stn_lr', lower=1e-6, upper=1., log=True, default_value=0.0001)
        cs.add_hyperparameter(stn_lr)

        # cond = InCondition(stn_wd, stn_optimizer, [True])
        # cs.add_condition(cond)

        cond = InCondition(stn_lr, stn_optimizer, [True])
        cs.add_condition(cond)

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='stn_warmup_epochs', choices=[0, 1, 2, 4, 6, 8, 10, 20], default_value=0))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='separate_localization_net', choices=[True, False], default_value=False))
        # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='use_unbounded_stn', choices=[True, False], default_value=True))

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='stn_conv1_depth', choices=[4, 8, 16, 32], default_value=8))
        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='stn_conv2_depth', choices=[4, 8, 16], default_value=8))

        # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='stn_theta_norm', choices=[True, False], default_value=True))

        penalty_loss = CSH.CategoricalHyperparameter(name='penalty_loss', choices=["OverlapPenalty", "ThetaLoss", "ThetaCropsPenalty"]+[""], default_value="")
        cs.add_hyperparameter(penalty_loss)

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='epsilon', choices=[0.01, 0.1, 1.0, 2., 3.], default_value=1.))
        
        # invert_penalty = CSH.CategoricalHyperparameter(name='invert_penalty', choices=[True, False], default_value=True)
        # cs.add_hyperparameter(invert_penalty)

        penalty_target = CSH.CategoricalHyperparameter(name='penalty_target', choices=['zero', 'one', 'mean', 'rand'], default_value="mean")
        cs.add_hyperparameter(penalty_target)

        cond = NotEqualsCondition(penalty_target, penalty_loss, "")
        cs.add_condition(cond)

        cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='penalty_weight', choices=[0.01, 0.1, 1.0, 2., 3.], default_value=1.))

        ema = CSH.CategoricalHyperparameter(name='stn_ema_update', choices=[True, False], default_value=False)
        cs.add_hyperparameter(ema)

        ema_momentum = CSH.CategoricalHyperparameter(name='stn_ema_momentum', choices=[0.5, 0.6, 0.7, 0.8, 0.9, 0.99, 0.998, 0.999], default_value=0.998)
        cs.add_hyperparameter(ema_momentum)

        cond = InCondition(ema_momentum, ema, [True])
        cs.add_condition(cond)

        min_glb_overlap = CSH.CategoricalHyperparameter(name='min_glb_overlap', choices=[0.1, 0.2, 0.4, 0.5, 0.6, 0.8, 0.9], default_value=0.5)
        cs.add_hyperparameter(min_glb_overlap)
        
        cond = EqualsCondition(min_glb_overlap, penalty_loss, 'OverlapPenalty')
        cs.add_condition(cond)

        # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='four_way_loss', choices=[True, False], default_value=False))

        global_crops_scale = CSH.CategoricalHyperparameter(name='global_crops_scale', choices=[(0.4, 1.), (0.6, 1.), (0.8, 1.), (0.9, 1.), (0.1, 0.8)], default_value=(0.4, 1.))
        cs.add_hyperparameter(global_crops_scale)

        cond = EqualsCondition(global_crops_scale, penalty_loss, 'OverlapPenalty')
        cs.add_condition(cond)

        # cs.add_hyperparameter(CSH.CategoricalHyperparameter(name='stn_mode', choices=["affine", "translation_scale_symmetric", "rotation_translation_scale_symmetric"], default_value="translation_scale_symmetric"))

        return cs

    def get_specific_config(self, cso, default_config, budget):
        config = deepcopy(default_config)
        return config


    def compute(self, config_id, config, budget, working_dir, default_config, **kwargs):
        print("--------------sampled config--------------")
        for k, v in config.items():
            print(f"{k}: {v}")
            
        config_id_formated = "_".join(map(str, config_id))
        set_args(default_config, config)

        working_expt_dir = pathlib.Path(f'{working_dir}/{config_id_formated}')
        default_config.expt_name = working_expt_dir
    
        print('----------------------------')
        print("START BOHB ITERATION")
        print('CONFIG: ' + str(default_config))
        print('BUDGET: ' + str(budget))
        print('----------------------------')

        info = {}
        info['config'] = str(default_config)

        stn_train_dct = {
            "epochs": 300,
            "use_pretrained_stn": False,
        }

        linear_eval_dct = {
            "lr": 0.1,
            "weight_decay": 0.0,
            "epochs": 90,
            "batch_size": 2048,
            "momentum": 0.9,
            "pretrained": f"{default_config.expt_name}/checkpoint_training_stn.pth.tar",
            "resume": f"{default_config.expt_name}/checkpoint_linear_eval.pth.tar",
        }

        # STN training
        if 'pretrain' in default_config.pipeline_mode:
            set_args(default_config, stn_train_dct)
            print("==> starting STN training.")
            try:
                run_simiam(default_config, working_expt_dir)
            except Exception as e:
                print(e)
                info["exception"] = str(e)
                print("==> STN training failed.")
                return {
                    "loss": float('inf'),
                    "info": info,
                }
            print("==> finished STN training.")

        # Linear evaluation
        if 'eval' in default_config.pipeline_mode:
            set_args(default_config, linear_eval_dct)

            if not os.path.isfile(default_config.pretrained):
                print("==> skipping linear eval because pretrained checkpoint does not exist.")
                return {
                    "loss": float('inf'),
                    "info": info,
                }
            print(f"==> starting linear eval with checkpoint {default_config.pretrained} (or checkpoint_linear_eval.pth.tar if exists).")
            try:
                score = run_linear(default_config, working_expt_dir)
            except Exception as e:
                print(e)
                info["exception"] = str(e)
                print("==> linear eval failed.")
                return {
                    "loss": float('inf'),
                    "info": info,
                }
            print("==> finished linear eval")
        

        print('----------------------------')
        print('FINAL SCORE: ' + str(-score))
        print("END BOHB ITERATION")
        print('----------------------------')

        return {
            "loss": -score,
            "info": info
        }


def set_args(outer_args, dct):
    for k, v in dct.items():
        if k in vars(outer_args):
            setattr(outer_args, k, v)
        else:
            print(f"{k} not in args. Not setting it.")


if __name__ == "__main__":
    outer_args = parser.parse_args()
    assert all([mode in ['pretrain', 'frozen', 'eval'] for mode in outer_args.pipeline_mode]), "pipeline_mode must be one of ['pretrain', 'frozen', 'eval']"

    x = datetime.datetime.now()
    working_dir = f'experiments/bohb_simsiam_{outer_args.run_id}'

    print(f"Working dir: {working_dir}")

    res = run_bohb_parallel(run_id=outer_args.run_id,
                            bohb_workers=outer_args.n_workers,
                            experiment_wrapper=ExperimentWrapper(),
                            working_dir=working_dir,
                            is_worker=outer_args.is_worker, 
                            default_config=outer_args)
