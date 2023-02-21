#!/usr/bin/env python
# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import argparse
import builtins
import math
import os
import random
import shutil
import time
import warnings
import pathlib

import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.distributed as dist
import torch.optim
import torch.multiprocessing as mp
import torch.utils.data
import torch.utils.data.distributed
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torchvision.models as models
from torch.utils.tensorboard import SummaryWriter
import yaml
import copy
import numpy as np
from stn import N_PARAMS

from models import resnet_cifar

import simsiam.loader
import simsiam.builder

import utils
import penalties
from utils import SummaryWriterCustom, custom_collate, get_stn_transforms, normalize_cifar10, normalize_imagenet
from stn import AugmentationNetwork, STN


model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))


penalty_list = sorted(name for name in penalties.__dict__
                      if name[0].isupper() and not name.startswith("__") and callable(penalties.__dict__[name]))
penalty_dict = {
    penalty: penalties.__dict__[penalty] for penalty in penalty_list
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
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
parser.add_argument('-b', '--batch-size', default=1024, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.03, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=0.0005, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
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

parser.add_argument('--dataset', default='s', type=str, help='dataset name. Should be one of ImageNet or CIFAR10.')
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
parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
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
parser.add_argument("--resize_all_inputs", default=False, type=utils.bool_flag,
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
parser.add_argument("--penalty_target", default='mean', type=str, choices=['zero', 'one', 'mean', 'rand'],
                        help="Specify the type of target of the penalty. Here, the target is the area with respect to"
                             "the original image. `zero` and `one` are the values itself. `mean` and `rand` are"
                             "inferred with respect to given crop-scales.")
parser.add_argument("--min_glb_overlap", default=0.5, type=float, help="The minimal overlap between the two global crops.")
parser.add_argument("--min_lcl_overlap", default=0.1, type=float, help="The minimal overlap between two local crops.")

parser.add_argument("--stn_ema_update", default=False, type=utils.bool_flag, help="")
parser.add_argument("--stn_ema_momentum", default=0.998, type=int, help="")

parser.add_argument('--use_pretrained_stn', default=False, type=utils.bool_flag, metavar='PATH', help='')

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


def main(kwargs=None, working_dir=None):
    args, unknown = parser.parse_known_args()
    # overload args with kwargs
    if kwargs is not None:
        for k, v in vars(kwargs).items():
            if k in vars(args):
                setattr(args, k, v)
            else:
                print(f"{k} not in args. Not setting it.")

    # Saving checkpoint and config pased on experiment mode
    # expt_dir = "experiments"
    # expt_sub_dir = os.path.join(expt_dir, args.expt_name)

    # args.expt_dir = pathlib.Path(expt_sub_dir)

    # if not os.path.exists(expt_sub_dir):
    #     os.makedirs(expt_sub_dir)

    expt_sub_dir = working_dir
    args.expt_dir = working_dir

    if not os.path.exists(args.expt_dir):
        os.makedirs(args.expt_dir)

    assert args.local_crops_number == 0, "SimSiam only uses two views, so local_crops_number must be 0"

    if args.stn_color_augment:
        print("setting stn_color_augment to True has no effect since we always color augment in SimSiam.")

    args_dict = vars(args)
    print(args_dict)
    timestr = time.strftime("%Y%m%d-%H%M%S")

    with open(os.path.join(expt_sub_dir, f"args_dict_pretraining_{timestr}.yaml"), "w") as f:
        yaml.dump(args_dict, f)

    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
        warnings.warn('You have chosen to seed training. '
                      'This will turn on the CUDNN deterministic setting, '
                      'which can slow down your training considerably! '
                      'You may see unexpected behavior when restarting '
                      'from checkpoints.')

    if args.gpu is not None:
        warnings.warn('You have chosen a specific GPU. This will completely '
                      'disable data parallelism.')

    if args.dist_url == "env://" and args.world_size == -1:
        args.world_size = int(os.environ["WORLD_SIZE"])

    args.distributed = args.world_size > 1 or args.multiprocessing_distributed

    ngpus_per_node = torch.cuda.device_count()
    print("ngpus per node", ngpus_per_node)
    try:
        if args.multiprocessing_distributed:
            # Since we have ngpus_per_node processes per node, the total world_size
            # needs to be adjusted accordingly
            args.world_size = ngpus_per_node * args.world_size
            # Use torch.multiprocessing.spawn to launch distributed processes: the
            # main_worker process function
            mp.spawn(main_worker, nprocs=ngpus_per_node, args=(ngpus_per_node, args))
        else:
            # Simply call main_worker function
            main_worker(args.gpu, ngpus_per_node, args)
    except Exception as e:
        raise e


def main_worker(gpu, ngpus_per_node, args):
    args.gpu = gpu

    # suppress printing if not master
    if args.multiprocessing_distributed and args.gpu != 0:
        def print_pass(*args):
            pass
        builtins.print = print_pass

    if args.gpu is not None:
        print("Use GPU: {} for training".format(args.gpu))

    if args.distributed:
        if args.dist_url == "env://" and args.rank == -1:
            args.rank = int(os.environ["RANK"])
        if args.multiprocessing_distributed:
            # For multiprocessing distributed training, rank needs to be the
            # global rank among all the processes
            args.rank = args.rank * ngpus_per_node + gpu
        dist.init_process_group(backend=args.dist_backend, init_method=args.dist_url,
                                world_size=args.world_size, rank=args.rank)
        torch.distributed.barrier()
    
    # create model
    if args.dataset == 'CIFAR10': 
        if args.theta_prediction_loss:
            theta_layer_dim = 6
        else:
            theta_layer_dim = None
        print(f"=> creating model {resnet_cifar.resnet18.__name__}")
        model = simsiam.builder.SimSiam(resnet_cifar.resnet18, args.dim, args.pred_dim, theta_layer_dim=theta_layer_dim)
    elif args.dataset == 'ImageNet':
        print(f"=> creating model {args.arch}")
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, 
            args.pred_dim)

    stn_ema_updater = None
    find_unused_parameters = False
    if args.stn_ema_update:
        stn_ema_updater = utils.EMA(beta=args.stn_ema_momentum)
        find_unused_parameters=True

    stn_penalty = None
    stn = None
    if args.use_stn:
        transform_net = STN(
        mode=args.stn_mode,
        invert_gradients=args.invert_stn_gradients,
        separate_localization_net=args.separate_localization_net,
        conv1_depth=args.stn_conv1_depth,
        conv2_depth=args.stn_conv2_depth,
        theta_norm=args.stn_theta_norm,
        local_crops_number=args.local_crops_number,
        global_crops_scale=args.global_crops_scale,
        local_crops_scale=args.local_crops_scale,
        resolution=args.stn_res,
        unbounded_stn=args.use_unbounded_stn,
        )
        stn = AugmentationNetwork(
        transform_net=transform_net,
        resize_input=args.resize_input,
        resize_size=args.resize_size,
        )
      
        if args.penalty_loss:
            stn_penalty = penalty_dict[args.penalty_loss](
                invert=args.invert_penalty,
                eps=args.epsilon,
                target=args.penalty_target,
                local_crops_scale=args.local_crops_scale,
                global_crops_scale=args.global_crops_scale,
                min_glb_overlap=args.min_glb_overlap,
                min_lcl_overlap=args.min_lcl_overlap,
                resolution=32,
                exponent=2,
                bins=100,
            ).cuda()

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    if args.use_stn:
        init_stn_lr = args.stn_lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
        if args.use_stn:
            stn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stn)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            if args.use_stn:
                stn.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            if args.use_stn:
                stn = torch.nn.parallel.DistributedDataParallel(stn, device_ids=[args.gpu], find_unused_parameters=find_unused_parameters)
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            if args.use_stn:
                stn = torch.nn.parallel.DistributedDataParallel(stn)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        if args.use_stn:
            stn = stn.cuda(args.gpu)

        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    
    target_stn = None
    if args.stn_ema_update and args.use_stn:
        target_stn = copy.deepcopy(stn)
        # there is no backpropagation through the stn target, so no need for gradients
        for p in target_stn.parameters():
            p.requires_grad = False
        
        print(target_stn)
    
    print(model) # print model after SyncBatchNorm
    print(stn)

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    stn_optimizer = None

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
        
        if args.theta_prediction_loss:
            optim_params += [{'params': model.module.theta_predictor.parameters(), 'fix_lr': False}]

        if not args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
            optim_params += [{'params': stn.parameters(), 'fix_lr': False}]
    else:
        optim_params = list(model.parameters())
        if not args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
            optim_params += list(stn.parameters())


    optimizer = torch.optim.SGD(optim_params, 
                                init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    if args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
        stn_optimizer = torch.optim.AdamW(params=list(stn.parameters()),
                                          lr=init_stn_lr,
                                          weight_decay=args.stn_weight_decay,
                                    )


    base_checkpoint_name = "checkpoint_frozen_stn" if args.use_pretrained_stn else "checkpoint_training_stn"
    checkpoint_to_resume = os.path.join(args.expt_dir, base_checkpoint_name + ".pth.tar")
    first_time_frozen_stn = False

    # if we want to train with frozen STN weights but no checkpoint of this run exists yet -> load stn training checkpoint
    if not os.path.isfile(checkpoint_to_resume):
        checkpoint_to_resume = os.path.join(args.expt_dir, "checkpoint_training_stn.pth.tar")
        first_time_frozen_stn = True
    
    # optionally resume from a checkpoint
    if os.path.isfile(checkpoint_to_resume):
        print("=> loading checkpoint '{}'".format(checkpoint_to_resume))
        if args.gpu is None:
            checkpoint = torch.load(checkpoint_to_resume)
        else:
            # Map model to be loaded to specified single gpu.
            loc = 'cuda:{}'.format(args.gpu)
            checkpoint = torch.load(checkpoint_to_resume, map_location=loc)
        
        model.load_state_dict(checkpoint['state_dict'])
        print(f"=> loaded backbone weights.")

        if args.use_stn:
            stn.load_state_dict(checkpoint['stn'])
            print(f"=> loaded stn weights.")

        # loading the old optimizer isn't possible since we optimize fewer parameters when running
        # frozen stn mode
        if not first_time_frozen_stn:
            args.start_epoch = checkpoint['epoch']
            optimizer.load_state_dict(checkpoint['optimizer'])
            print(f"=> loaded optimizer.")

        if stn_optimizer:
            stn_optimizer.load_state_dict(checkpoint['stn_optimizer'])
            print(f"=> loaded stn optimizer.")
        
        if not args.use_pretrained_stn:
            print(f"=> loaded checkpoint (epoch {checkpoint['epoch']}).")

        if args.use_pretrained_stn and args.use_stn:
            for p in stn.parameters():
                p.requires_grad = False
            print(f"=> stn weights frozen.")
            if first_time_frozen_stn:
                    args.start_epoch = 0
    else:
        if args.use_pretrained_stn:
            print("=> no checkpoint found to resume from. Starting fresh training with frozen STN weights.")
        else:
            print("=> no checkpoint found to resume from. Starting fresh STN training.")
        
    cudnn.benchmark = True
    summary_writer = None
    if args.rank == 0:
        summary_writer = SummaryWriterCustom(args.expt_dir / "summary", plot_size=args.summary_plot_size)

    if not args.resize_all_inputs and args.dataset == 'ImageNet':
        print("using ImageNet transforms (resize_all_inputs)")
        transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_imagenet,
            ])
        collate_fn = custom_collate

    elif args.resize_all_inputs and args.dataset == "ImageNet":
        # Data loading code
        traindir = os.path.join(args.dataset_path, 'train')
        print("using ImageNet transforms")

        transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                transforms.RandomGrayscale(p=0.2),
                transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
                utils.NoneTransform() if args.use_stn else transforms.RandomHorizontalFlip(), #removed RandomHorizontalFlip as it is learned by the STN
                transforms.ToTensor(),
                normalize_imagenet,
            ])

        collate_fn = None
        if args.use_stn:
            train_dataset = datasets.ImageFolder(traindir, transform=transform)
        else:
            train_dataset = datasets.ImageFolder(traindir, simsiam.loader.TwoCropsTransform(transform))
    
    elif args.dataset == 'CIFAR10':
        
        if args.use_stn:
            print("using CIFAR10 STN transforms")
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.ToTensor(),
                ])
        else:
            print("using CIFAR10 default transforms")
            transform = transforms.Compose([
                    transforms.RandomResizedCrop(32, scale=(0.2, 1.)),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),  
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_cifar10,
                ])


        collate_fn = None
        if args.use_stn:
            train_dataset = datasets.CIFAR10(root='datasets/CIFAR10', train=True, download=True, transform=transform)
        else:
            train_dataset = datasets.CIFAR10(root='datasets/CIFAR10', train=True, download=True, transform=simsiam.loader.TwoCropsTransform(transform))
    else:
        raise ValueError("Dataset not supported.")

    print(f"Data loaded: there are {len(train_dataset)} images.")

    if args.distributed:
        train_sampler = torch.utils.data.distributed.DistributedSampler(train_dataset)
    else:
        train_sampler = None

    train_loader = torch.utils.data.DataLoader(
        train_dataset, 
        batch_size=args.batch_size, 
        shuffle=(train_sampler is None),
        num_workers=args.workers, 
        pin_memory=True, 
        sampler=train_sampler, 
        drop_last=False, 
        collate_fn=collate_fn
        )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)

        if args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
            adjust_learning_rate(stn_optimizer, init_stn_lr, epoch, args)

        if args.stn_warmup_epochs > 0:
            if epoch < args.stn_warmup_epochs:
                for p in stn.parameters():
                    p.requires_grad = False
            else:
                for p in stn.parameters():
                    p.requires_grad = True

        try:
            # train for one epoch
            train(train_loader, model, criterion, optimizer, stn_optimizer, stn, target_stn, stn_ema_updater, epoch, args, summary_writer, stn_penalty)
        except Exception as e:
            raise e
            
        is_last_epoch = epoch + 1 >= args.epochs

        save_dict = {
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                    'stn': stn.state_dict() if args.use_stn else None,
                    'stn_optimizer': stn_optimizer.state_dict() if args.use_stn and stn_optimizer else None,
                }

        checkpoint_filename_epoch_wise = os.path.join(args.expt_dir, base_checkpoint_name + f'_{epoch:04d}.pth.tar')

        if args.rank == 0:
            save_checkpoint(save_dict, is_best=False, filename=os.path.join(args.expt_dir, base_checkpoint_name + ".pth.tar"))
            #save_checkpoint(save_dict, is_best=False, filename=os.path.join(args.expt_dir, "model.pth.tar"))

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank == 0):
            if epoch % args.save_freq == 0 or is_last_epoch:
                save_checkpoint(save_dict, is_best=False, filename=checkpoint_filename_epoch_wise)


    # shut down the writer at the end of training
    if summary_writer and args.rank == 0:
        summary_writer.close()


def train(train_loader, model, criterion, optimizer, stn_optimizer, stn, target_stn, stn_ema_updater, epoch, args, summary_writer=None, stn_penalty=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    simsiam_loss = AverageMeter('SimSiam Loss', ':.4f')
    total_loss = AverageMeter('Total Loss', ':.4f')
    meters = [batch_time, data_time, simsiam_loss]

    if stn_penalty:
       penalty_loss_meter = AverageMeter('Penalty Loss', ':.6f')
       meters.append(penalty_loss_meter)    

    meters.append(total_loss)
    
    progress = ProgressMeter(
        len(train_loader),
        meters,
        prefix="Epoch: [{}/{}]".format(epoch, args.epochs-1))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            if args.use_stn:
                images = images.cuda(args.gpu, non_blocking=True)
            else:
                images[0] = images[0].cuda(args.gpu, non_blocking=True)
                images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if args.use_stn:
            stn_images, thetas = stn(images)
            if args.stn_ema_update:
                with torch.no_grad():
                    stn_images_target, thetas_target = target_stn(images)
                    thetas[1] = thetas_target[1]
                    stn_images[1] = stn_images_target[1]
        else:
            stn_images = images

        penalty = None
        if args.use_stn:
            if stn_penalty:
                if args.penalty_loss == 'ThetaCropsPenalty':
                    penalty = torch.tensor(0.0).cuda(args.gpu)
                    for t in thetas:
                        penalty += stn_penalty(thetas=t, crops_scale=args.global_crops_scale)
                    penalty = penalty.mean()
                else:
                    penalty = stn_penalty(images=stn_images, target=images, thetas=thetas)
        if args.use_stn:        
            transform_view1, transform_view2 = get_stn_transforms(args)
            stn_images[0] = transform_view1(stn_images[0])
            stn_images[1] = transform_view2(stn_images[1])   
            # TODO: do this for stn_images_target and check validity for ImageNet    
        
        # compute output and loss
        penalty_l = 0
        if args.use_stn and penalty:
            penalty_l = torch.exp(penalty * args.penalty_weight)
        
        if args.four_way_loss:
            p1, p2, z1, z2 = model(x1=images, x2=stn_images[0])
            p3, p4, z3, z4 = model(x1=images, x2=stn_images[1])
            simsiam_l1 = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.25
            simsiam_l2 = -(criterion(p3, z3).mean() + criterion(p4, z4).mean()) * 0.25

            simsiam_l = simsiam_l1 + simsiam_l2
            total_l = simsiam_l + penalty_l

        elif args.theta_prediction_loss:
            theta1_pred = model(x1=images, x2=stn_images[0])
            theta2_pred = model(x1=images, x2=stn_images[1])
            simsiam_l1 = nn.functional.mse_loss(theta1_pred, thetas[0].flatten(start_dim=1))
            simsiam_l2 = nn.functional.mse_loss(theta2_pred, thetas[1].flatten(start_dim=1))
            simsiam_l = simsiam_l1 + simsiam_l2
            total_l = simsiam_l + penalty_l
        
        else:
            p1, p2, z1, z2 = model(x1=stn_images[0], x2=stn_images[1])
            
            simsiam_l = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5
            total_l = simsiam_l + penalty_l

        if math.isnan(float(total_l.item())):
            raise ValueError('Loss is NaN')

        total_loss.update(total_l.item(), images[0].size(0))
        simsiam_loss.update(simsiam_l.item(), images[0].size(0))
        
        if stn_penalty and args.use_stn and penalty:
            penalty_loss_meter.update(penalty_l.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()

        if args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
            stn_optimizer.zero_grad()

        total_l.backward()
        optimizer.step()

        # print(model.module.encoder.conv3_x[0].residual_function[0].weight.grad)

        if args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
            stn_optimizer.step()

        if args.stn_ema_update and args.use_stn and epoch >= args.stn_warmup_epochs:
            utils.update_moving_average(stn_ema_updater, ma_model=target_stn, current_model=stn)

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()
        
        # write log
        if args.rank == 0:
            tag_loss = "Total Loss (frozen STN)" if args.use_pretrained_stn else "Total Loss (training STN)"
            tag_loss_simsiam = "SimSiam Loss (frozen STN)" if args.use_pretrained_stn else "SimSiam Loss (training STN)"
            tag_lr = "lr (frozen STN)" if args.use_pretrained_stn else "lr (training STN)"
            summary_writer.write_scalar(tag=tag_loss, scalar_value=total_l.item(), epoch=epoch+1)
            summary_writer.write_scalar(tag=tag_loss_simsiam, scalar_value=simsiam_l.item(), epoch=epoch+1)
            summary_writer.write_scalar(tag=tag_lr, scalar_value=optimizer.param_groups[0]["lr"], epoch=epoch+1)
            
            if stn_penalty and not args.use_pretrained_stn and args.use_stn:
                summary_writer.write_scalar(tag="Penalty Loss", scalar_value=penalty.item(), epoch=epoch+1)

            if args.use_stn_optimizer and not args.use_pretrained_stn and args.use_stn:
                summary_writer.write_scalar(tag="lr stn", scalar_value=stn_optimizer.param_groups[0]["lr"], epoch=epoch+1)

            if epoch % args.grad_check_freq == 0 and i == 0 and args.use_stn:
                utils.print_gradients(stn, args)
            
            if i % args.print_freq == 0:
                progress.display(i)

            # Log stuff to tensorboard
            if epoch % args.summary_writer_freq == 0 and i == 0 and args.use_stn:
                tag_images = "images (frozen STN)" if args.use_pretrained_stn else "images (training STN)"
                tag_thetas = "theta (frozen STN)" if args.use_pretrained_stn else "theta (training STN)"
                summary_writer.write_stn_info(stn_images, images, thetas, epoch+1, tag_images=tag_images, tag_thetas=tag_thetas)


def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self, name, fmt=':f'):
        self.name = name
        self.fmt = fmt
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count

    def __str__(self):
        fmtstr = '{name} {val' + self.fmt + '} ({avg' + self.fmt + '})'
        return fmtstr.format(**self.__dict__)


class ProgressMeter(object):
    def __init__(self, num_batches, meters, prefix=""):
        self.batch_fmtstr = self._get_batch_fmtstr(num_batches)
        self.meters = meters
        self.prefix = prefix

    def display(self, batch):
        entries = [self.prefix + self.batch_fmtstr.format(batch)]
        entries += [str(meter) for meter in self.meters]
        print('\t'.join(entries))

    def _get_batch_fmtstr(self, num_batches):
        num_digits = len(str(num_batches // 1))
        fmt = '{:' + str(num_digits) + 'd}'
        return '[' + fmt + '/' + fmt.format(num_batches) + ']'


def adjust_learning_rate(optimizer, init_lr, epoch, args):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / args.epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr


if __name__ == '__main__':
    main()
