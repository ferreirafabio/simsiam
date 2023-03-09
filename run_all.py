from main_simsiam import main as run_simiam
from main_lincls import main as run_linear
import torchvision.models as models
import argparse
import utils
import time
import os
import penalties

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
parser.add_argument('-b', '--batch-size', default=2048, type=int,
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
                    
parser.add_argument('--stn-wd', '--stn-weight-decay', default=0.0005, type=float,
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
parser.add_argument('--save-freq', default=50, type=int, metavar='N', help='checkpoint save frequency (default: 10)')

# STN
parser.add_argument("--invert_stn_gradients", default=True, type=utils.bool_flag,
                    help="Set this flag to invert the gradients used to learn the STN")
parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate optimizer for the STN parameters; "
                            "annealed with cosine and no warmup")
parser.add_argument('--stn_mode', default='affine', type=str,
                    help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                            'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
parser.add_argument("--stn_lr", default=5e-5, type=float, help="""Learning rate at the end of
                    linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
                    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate localization network for each head.")
parser.add_argument("--summary_writer_freq", default=25, type=int, 
help="Defines the number of iterations the summary writer will write output.")
parser.add_argument("--grad_check_freq", default=50, type=int,
                    help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
parser.add_argument("--summary_plot_size", default=16, type=int,
                    help="Defines the number of samples to show in the summary writer.")
parser.add_argument('--stn_pretrained_weights', default='', type=str,
                    help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
parser.add_argument("--deep_loc_net", default=False, type=utils.bool_flag,
                    help="(legacy) Set this flag to use a deep loc net. (default: False).")
parser.add_argument("--stn_res", default=(32, 32), type=int, nargs='+',
                    help="Set the resolution of the global and local crops of the STN (default: 32x and 32x)")
parser.add_argument("--use_unbounded_stn", default=False, type=utils.bool_flag,
                    help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                    help="Specifies the number of warmup epochs for the STN (default: 0).")
parser.add_argument("--stn_conv1_depth", default=16, type=int,
                    help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
parser.add_argument("--stn_conv2_depth", default=16, type=int,
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
parser.add_argument("--min_glb_overlap", default=0.5, type=float, help="The minimal overlap between the two global crops.")
parser.add_argument("--min_lcl_overlap", default=0.1, type=float, help="The minimal overlap between two local crops.")

parser.add_argument('--pretrained', default='', type=str,
                    help='path to simsiam pretrained checkpoint')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('--use_pretrained_stn', default=False, type=utils.bool_flag, metavar='PATH',
                    help='')

parser.add_argument('--four_way_loss', default=False, type=utils.bool_flag, help='')
parser.add_argument('--four_way_loss2', default=False, type=utils.bool_flag, help='')
parser.add_argument('--theta_prediction_loss', default=False, type=utils.bool_flag, help='')
parser.add_argument('--use_stn', default=True, type=utils.bool_flag, help='for testing reasons')
parser.add_argument('--adco', default=False, type=utils.bool_flag, help='Train one localization net and head adversarilly and the other cooperativelly.')


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
parser.add_argument('--view_reconstruction_loss', default=False, type=utils.bool_flag, help='')
parser.add_argument('--view_reconstruction_loss2', default=False, type=utils.bool_flag, help='')
parser.add_argument("--finetune", default=False, type=utils.bool_flag, help="Finetune whole network instead of linear eval protocol.")

def set_args(outer_args, dct):
    for k, v in dct.items():
        if k in vars(outer_args):
            setattr(outer_args, k, v)
        else:
            print(f"{k} not in args. Not setting it.")


if __name__ == '__main__':
        outer_args = parser.parse_args()

        if outer_args.adco:
            assert outer_args.separate_localization_net == True, "if adco is set, separate_localization_net must be set to True."
            
        assert all([mode in ['pretrain', 'frozen', 'eval'] for mode in outer_args.pipeline_mode]), "pipeline_mode must be one of ['pretrain', 'frozen', 'eval']"

        print(f"running pipeline mode: {outer_args.pipeline_mode}")

        if outer_args.epochs is not None:
            epochs = outer_args.epochs
        elif "frozen" in outer_args.pipeline_mode:
            epochs = 100
        else:   
            epochs = 800

        stn_train_dct = {
            "epochs": epochs,
            "use_pretrained_stn": False,
        }

        frozen_stn_dct = {
            "epochs": 800,
            "use_pretrained_stn": True,
        }

        linear_eval_dct = {
            "lr": 0.1,
            "weight_decay": 0.0,
            "epochs": 90,
            "batch_size": 2048,
            "momentum": 0.9,
            "pretrained": f"experiments/{outer_args.expt_name}/checkpoint_frozen_stn.pth.tar" if "frozen" in outer_args.pipeline_mode else f"experiments/{outer_args.expt_name}/checkpoint_training_stn.pth.tar",
            "resume": f"experiments/{outer_args.expt_name}/checkpoint_linear_eval.pth.tar",
        }

        # STN training
        if 'pretrain' in outer_args.pipeline_mode:
            set_args(outer_args, stn_train_dct)
            print("==> starting STN training.")
            run_simiam(outer_args)
            print("==> finished STN training.")
        
            time.sleep(10)

        # Frozen STN, standard pre-training
        if 'frozen' in outer_args.pipeline_mode:
            set_args(outer_args, frozen_stn_dct)
            print("==> starting standard pre-training with frozen STN.")
            run_simiam(outer_args)
            print("==> finished pre-training with frozen STN")
            
            time.sleep(10)


        # Run linear evaluation
        if 'eval' in outer_args.pipeline_mode:
            set_args(outer_args, linear_eval_dct)
            print(f"==> starting linear eval with checkpoint {outer_args.pretrained} (or checkpoint_linear_eval.pth.tar if exists).")
            run_linear(outer_args)
            print("==> finished linear eval")
