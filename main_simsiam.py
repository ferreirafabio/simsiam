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

from models import resnet_cifar

import simsiam.loader
import simsiam.builder

import utils
from utils import SummaryWriterCustom, custom_collate
from penalty_losses import HISTLoss, SIMLoss, ThetaLoss, ThetaCropsPenalty
from stn import AugmentationNetwork, STN

normalize_imagenet = transforms.Normalize(
    mean=[0.485, 0.456, 0.406],
    std=[0.229, 0.224, 0.225]
)

normalize_cifar10 = transforms.Normalize(
    mean=[0.4914, 0.4822, 0.4465],
    std=[0.2023, 0.1994, 0.2010]
)

model_names = sorted(name for name in models.__dict__
    if name.islower() and not name.startswith("__")
    and callable(models.__dict__[name]))

penalty_dict = {
    "simloss": SIMLoss,
    "histloss": HISTLoss,
    "thetaloss": ThetaLoss,
    "thetacropspenalty": ThetaCropsPenalty,
}

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument("--dataset_path", type=str, default="/data/datasets/ImageNet/imagenet-pytorch", help="path to dataset")
parser.add_argument('-a', '--arch', metavar='ARCH', default='resnet50',
                    choices=model_names,
                    help='model architecture: ' +
                        ' | '.join(model_names) +
                        ' (default: resnet50)')
parser.add_argument('-j', '--workers', default=32, type=int, metavar='N',
                    help='number of data loading workers (default: 32)')
parser.add_argument('--epochs', default=100, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('-b', '--batch-size', default=512, type=int,
                    metavar='N',
                    help='mini-batch size (default: 512), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=10, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
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

parser.add_argument('--dataset', default='ImageNet', type=str, help='dataset name. Should be one of ImageNet or CIFAR10.')
parser.add_argument('--expt-name', default='default_experiment', type=str, help='Name of the experiment')
parser.add_argument('--save-freq', default=10, type=int, metavar='N', help='checkpoint save frequency (default: 10)')

# STN
parser.add_argument("--invert_stn_gradients", default=False, type=utils.bool_flag,
                    help="Set this flag to invert the gradients used to learn the STN")
parser.add_argument("--use_stn_optimizer", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate optimizer for the STN parameters; "
                            "annealed with cosine and no warmup")
parser.add_argument('--stn_mode', default='affine', type=str,
                    help='Determines the STN mode (choose from: affine, translation, scale, rotation, '
                            'rotation_scale, translation_scale, rotation_translation, rotation_translation_scale')
parser.add_argument("--stn_lr", default=5e-4, type=float, help="""Learning rate at the end of
                    linear warmup (highest LR used during training) of the STN optimizer. The learning rate is linearly scaled
                    with the batch size, and specified here for a reference batch size of 256.""")
parser.add_argument("--separate_localization_net", default=False, type=utils.bool_flag,
                    help="Set this flag to use a separate localization network for each head.")
parser.add_argument("--summary_writer_freq", default=5000, type=int, 
help="Defines the number of iterations the summary writer will write output.")
parser.add_argument("--grad_check_freq", default=5000, type=int,
                    help="Defines the number of iterations the current tensor grad of the global 1 localization head is printed to stdout.")
parser.add_argument("--summary_plot_size", default=16, type=int,
                    help="Defines the number of samples to show in the summary writer.")
parser.add_argument('--stn_pretrained_weights', default='', type=str,
                    help="Path to pretrained weights of the STN network. If specified, the STN is not trained and used to pre-process images solely.")
parser.add_argument("--deep_loc_net", default=False, type=utils.bool_flag,
                    help="(legacy) Set this flag to use a deep loc net. (default: False).")
parser.add_argument("--stn_res", default=(224, 96), type=int, nargs='+',
                    help="Set the resolution of the global and local crops of the STN (default: 224x and 96x)")
parser.add_argument("--use_unbounded_stn", default=False, type=utils.bool_flag,
                    help="Set this flag to not use a tanh in the last STN layer (default: use bounded STN).")
parser.add_argument("--stn_warmup_epochs", default=0, type=int,
                    help="Specifies the number of warmup epochs for the STN (default: 0).")
parser.add_argument("--stn_conv1_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv1 for the STN localization network (default: 32).")
parser.add_argument("--stn_conv2_depth", default=32, type=int,
                    help="Specifies the number of feature maps of conv2 for the STN localization network (default: 32).")
parser.add_argument("--stn_theta_norm", default=False, type=utils.bool_flag,
                    help="Set this flag to normalize 'theta' in the STN before passing to affine_grid(theta, ...). Fixes the problem with cropping of the images (black regions)")
parser.add_argument("--use_stn_penalty", default=False, type=utils.bool_flag,
                    help="Set this flag to add a penalty term to the loss. Similarity between input and output image of STN.")
parser.add_argument("--penalty_loss", default="simloss", type=str, choices=list(penalty_dict.keys()),
                    help="Specify the name of the similarity to use.")
parser.add_argument("--epsilon", default=1., type=float,
                    help="Scalar for the penalty loss")
parser.add_argument("--invert_penalty", default=False, type=utils.bool_flag,
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

# simsiam specific configs:
parser.add_argument('--dim', default=2048, type=int,
                    help='feature dimension (default: 2048)')
parser.add_argument('--pred-dim', default=512, type=int,
                    help='hidden dimension of the predictor (default: 512)')
parser.add_argument('--fix-pred-lr', action='store_true',
                    help='Fix learning rate for the predictor')


def main():
    args = parser.parse_args()

    # Saving checkpoint and config pased on experiment mode
    expt_dir = "experiments"
    expt_sub_dir = os.path.join(expt_dir, args.expt_name)

    args.expt_dir = pathlib.Path(expt_sub_dir)

    if not os.path.exists(expt_sub_dir):
        os.makedirs(expt_sub_dir)

    assert args.local_crops_number == 0, "SimSiam only uses two views, so local_crops_number must be 0"

    if args.dataset == 'CIFAR10':
        args.epochs = 800
        args.lr = 0.03
        #args.batch_size = 512
        args.batch_size = 2048
        args.workers = 4
        args.weight_decay = 0.0005
        print(f"Changed hyperparameters for CIFAR10")

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
        print(f"=> creating model {resnet_cifar.resnet18.__name__}")
        model = simsiam.builder.SimSiam(resnet_cifar.resnet18, args.dim, args.pred_dim)
    elif args.dataset == 'ImageNet':
        print(f"=> creating model {args.arch}")
        model = simsiam.builder.SimSiam(
            models.__dict__[args.arch],
            args.dim, 
            args.pred_dim)

    #transform_net = STN(
    #    mode=args.stn_mode,
    #    invert_gradients=args.invert_stn_gradients,
    #    separate_localization_net=args.separate_localization_net,
    #    conv1_depth=args.stn_conv1_depth,
    #    conv2_depth=args.stn_conv2_depth,
    #    theta_norm=args.stn_theta_norm,
    #    local_crops_number=args.local_crops_number,
    #    global_crops_scale=args.global_crops_scale,
    #    local_crops_scale=args.local_crops_scale,
    #    resolution=args.stn_res,
    #    unbounded_stn=args.use_unbounded_stn,
    #)
    #stn = AugmentationNetwork(
    #    transform_net=transform_net,
    #    resize_input=args.resize_input,
    #    resize_size=args.resize_size,
    #)

    # infer learning rate before changing batch size
    init_lr = args.lr * args.batch_size / 256
    init_stn_lr = args.stn_lr * args.batch_size / 256

    if args.distributed:
        # Apply SyncBN
        model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    
        # TODO uncomment
        # stn = torch.nn.SyncBatchNorm.convert_sync_batchnorm(stn)

        # For multiprocessing distributed, DistributedDataParallel constructor
        # should always set the single device scope, otherwise,
        # DistributedDataParallel will use all available devices.
        if args.gpu is not None:
            torch.cuda.set_device(args.gpu)
            model.cuda(args.gpu)
            
            # TODO uncomment: 
            # stn.cuda(args.gpu)

            # When using a single GPU per process and per
            # DistributedDataParallel, we need to divide the batch size
            # ourselves based on the total number of GPUs we have
            args.batch_size = int(args.batch_size / ngpus_per_node)
            args.workers = int((args.workers + ngpus_per_node - 1) / ngpus_per_node)
            model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[args.gpu])
            # TODO uncomment
            # stn = torch.nn.parallel.DistributedDataParallel(stn, device_ids=[args.gpu])
        else:
            model.cuda()
            # DistributedDataParallel will divide and allocate batch_size to all
            # available GPUs if device_ids are not set
            model = torch.nn.parallel.DistributedDataParallel(model)
            
            # TODO uncomment: 
            # stn = torch.nn.parallel.DistributedDataParallel(stn)

    elif args.gpu is not None:
        torch.cuda.set_device(args.gpu)
        model = model.cuda(args.gpu)
        # TODO uncomment: 
        # stn = stn.cuda(args.gpu)

        # comment out the following line for debugging
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    else:
        # AllGather implementation (batch shuffle, queue update, etc.) in
        # this code only supports DistributedDataParallel.
        raise NotImplementedError("Only DistributedDataParallel is supported.")
    print(model) # print model after SyncBatchNorm
    
    # TODO uncomment: 
    # print(stn)

    # define loss function (criterion) and optimizer
    criterion = nn.CosineSimilarity(dim=1).cuda(args.gpu)

    # TODO uncomment:
    #stn_optimizer = None
    #use_pretrained_stn = os.path.isfile(args.stn_pretrained_weights)
    #if use_pretrained_stn:
        #state_dict = torch.load(args.stn_pretrained_weights, map_location="cpu")
        #msg = stn.load_state_dict(state_dict, strict=False)
        #print(f'Pretrained weights found at {args.stn_pretrained_weights} and loaded with msg: {msg}')
    
    #    for p in stn.parameters():
    #        p.requires_grad = False

    if args.fix_pred_lr:
        optim_params = [{'params': model.module.encoder.parameters(), 'fix_lr': False},
                        {'params': model.module.predictor.parameters(), 'fix_lr': True}]
    else:
        optim_params = model.parameters()

    # TODO uncomment:
    #if not args.use_stn_optimizer and not use_pretrained_stn:
        # optim_params += list(stn.parameters())

    optimizer = torch.optim.SGD(optim_params, init_lr,
                                momentum=args.momentum,
                                weight_decay=args.weight_decay)

    # TODO uncomment:
    #if args.use_stn_optimizer and not use_pretrained_stn:
    #    stn_optimizer = torch.optim.AdamW(list(stn.parameters()), lr=args.init_stn_lr)

    
    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            if args.gpu is None:
                checkpoint = torch.load(args.resume)
            else:
                # Map model to be loaded to specified single gpu.
                loc = 'cuda:{}'.format(args.gpu)
                checkpoint = torch.load(args.resume, map_location=loc)
            args.start_epoch = checkpoint['epoch']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    cudnn.benchmark = True
    writer = None
    summary_writer = None
    if args.rank == 0:
        writer = SummaryWriter(log_dir=os.path.join(args.expt_dir, f"tensorboard_pretraining_{args.epochs}_{init_lr}"))
        # TODO remove above
        summary_writer = SummaryWriterCustom(args.expt_dir / "summary", batch_size=args.batch_size)


    if not args.resize_all_inputs and args.dataset == 'ImageNet':
        transform = transforms.Compose([
                transforms.ToTensor(),
                normalize_imagenet,
            ])
        collate_fn = custom_collate

    elif args.resize_all_inputs and args.dataset == "ImageNet":
        # Data loading code
        traindir = os.path.join(args.dataset_path, 'train')

        # MoCo v2's aug: similar to SimCLR https://arxiv.org/abs/2002.05709
        transform = [
            transforms.RandomResizedCrop(224, scale=(0.2, 1.)),
            transforms.RandomApply([
                transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)  # not strengthened
            ], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            normalize_imagenet,
        ]

        # TODO uncomment:
        #transform = transforms.Compose([
                #transforms.Resize(256),
                #transforms.CenterCrop(224),
                #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                #transforms.RandomGrayscale(p=0.2),
                #transforms.RandomApply([simsiam.loader.GaussianBlur([.1, 2.])], p=0.5),
                # #removed RandomHorizontalFlip as it is learned by the STN
                #transforms.ToTensor(),
                #normalize_imagenet,
            #])

        collate_fn = None
        # TODO: remove TwoCropsTransform
        train_dataset = datasets.ImageFolder(traindir, simsiam.loader.TwoCropsTransform(transforms.Compose(transform)))
    
    elif args.dataset == 'CIFAR10':
        transform = simsiam.loader.TwoCropsTransform(transforms.Compose(
                [
                    transforms.RandomResizedCrop(size=32, scale=(0.2, 1.)),
                    transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                    transforms.RandomGrayscale(p=0.2),
                    transforms.RandomHorizontalFlip(),
                    transforms.ToTensor(),
                    normalize_cifar10,
                ]
            )
        )

        # TODO uncomment:
        #transform = transforms.Compose([
                #transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
                #transforms.RandomGrayscale(p=0.2),  
                # #removed RandomHorizontalFlip as it is learned by the STN
                #transforms.ToTensor(),
                #transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #])

        collate_fn = None
        train_dataset = datasets.CIFAR10(root='datasets/CIFAR10', train=True, download=True, transform=transform)
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
        drop_last=True, 
        collate_fn=collate_fn
        )

    for epoch in range(args.start_epoch, args.epochs):
        if args.distributed:
            train_sampler.set_epoch(epoch)
        adjust_learning_rate(optimizer, init_lr, epoch, args)
        # TODO uncomment:
        #if args.use_stn_optimizer and not use_pretrained_stn:
            #adjust_learning_rate(stn_optimizer, init_lr_stn, epoch, args)

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, args, writer)
        # TODO uncomment:
        #train(train_loader, model, criterion, optimizer, stn_optimizer, epoch, args, writer)

        is_last_epoch = epoch + 1 >= args.epochs

        if not args.multiprocessing_distributed or (args.multiprocessing_distributed and args.rank % ngpus_per_node == 0):
            if epoch % args.save_freq == 0 or is_last_epoch:
                save_checkpoint({
                    'epoch': epoch + 1,
                    'arch': args.arch,
                    'state_dict': model.state_dict(),
                    'optimizer' : optimizer.state_dict(),
                }, is_best=False, filename=os.path.join(args.expt_dir, 'checkpoint_{:04d}.pth.tar'.format(epoch)))


    # shut down the writer at the end of training
    if writer is not None and args.rank == 0:
        writer.close()


def train(train_loader, model, criterion, optimizer, epoch, args, writer=None):
    batch_time = AverageMeter('Time', ':6.3f')
    data_time = AverageMeter('Data', ':6.3f')
    losses = AverageMeter('Loss', ':.4f')
    progress = ProgressMeter(
        len(train_loader),
        [batch_time, data_time, losses],
        prefix="Epoch: [{}]".format(epoch))

    # switch to train mode
    model.train()

    end = time.time()
    for i, (images, _) in enumerate(train_loader):
        # measure data loading time
        data_time.update(time.time() - end)

        if args.gpu is not None:
            images[0] = images[0].cuda(args.gpu, non_blocking=True)
            images[1] = images[1].cuda(args.gpu, non_blocking=True)

        if i % args.summary_writer_freq == 0 and args.rank == 0:
            original_images = copy.deepcopy(images)

        #TODO uncomment:
        #images = stn(images)

        # if not args.resize_all_inputs and args.dataset == 'ImageNet':
            # # in the case of varying image resolutions, we could not color-augment images in DataLoader -> do it here:
            # color_jitter = transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1)
            # gaussian_blur = simsiam.loader.GaussianBlur([.1, 2.])
            # transform_view1 = transforms.Compose([
            #   transforms.RandomApply([color_jitter], p=0.8),
            #   transforms.RandomGrayscale(p=0.2),
            #   transforms.RandomApply([gaussian_blur)], p=0.5),
            #   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #   transforms.ConvertImageDtype(torch.float32),
            # ]                 
            # transform_view2 = transforms.Compose([
            #   transforms.RandomApply([color_jitter], p=0.8),
            #   transforms.RandomGrayscale(p=0.2),
            #   transforms.RandomApply([gaussian_blur)], p=0.5),
            #   transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
            #   transforms.ConvertImageDtype(torch.float32),
            # ]              
            # 
            # images[0] = transform_view1(images[0])
            # images[1] = transform_view2(images[1])          


        #if i % args.summary_writer_freq == 0 and args.rank == 0:
            #summary_writer.write_image_grid(tag="images", images=images, original_images=original_images, epoch=epoch, global_step=it)
            #summary_writer.write_theta_heatmap(tag="theta_g1", theta=stn.module.transform_net.affine_matrix_g1, epoch=epoch, global_step=it)
            #summary_writer.write_theta_heatmap(tag="theta_g2", theta=stn.module.transform_net.affine_matrix_g2, epoch=epoch, global_step=it)
            #summary_writer.write_theta_heatmap(tag="theta_l1", theta=stn.module.transform_net.affine_matrix_l1, epoch=epoch, global_step=it)
            #summary_writer.write_theta_heatmap(tag="theta_l2", theta=stn.module.transform_net.affine_matrix_l2, epoch=epoch, global_step=it)

            #theta_g_euc_norm = np.linalg.norm(np.double(stn.module.transform_net.affine_matrix_g2[0] - stn.module.transform_net.affine_matrix_g1[0]), 2)
            #theta_l_euc_norm = np.linalg.norm(np.double(stn.module.transform_net.affine_matrix_l2[0] - stn.module.transform_net.affine_matrix_l1[0]), 2)
            #summary_writer.write_scalar(tag="theta local eucl. norm.", scalar_value=theta_l_euc_norm, global_step=it)
            #summary_writer.write_scalar(tag="theta global eucl. norm.", scalar_value=theta_g_euc_norm, global_step=it)

            # original_images = None
            

        # compute output and loss
        p1, p2, z1, z2 = model(x1=images[0], x2=images[1])
        loss = -(criterion(p1, z2).mean() + criterion(p2, z1).mean()) * 0.5

        losses.update(loss.item(), images[0].size(0))

        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # TODO uncomment:
        #if args.use_stn_optimizer and not use_pretrained_stn:
        #   stn_optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        if args.rank == 0 and i % args.print_freq == 0:
            progress.display(i)
        
        # write log epoch-wise
        if args.rank == 0:
            writer.add_scalar('Pre-training/Loss', loss.item(), epoch + 1)
            # TODO uncomment:
            #summary_writer.write_scalar(tag="loss", scalar_value=loss.item(), global_step=it)
            #summary_writer.write_scalar(tag="lr", scalar_value=optimizer.param_groups[0]["lr"], global_step=it)

            #if args.use_stn_optimizer and not use_pretrained_stn:
            #   summary_writer.write_scalar(tag="stn lr", scalar_value=stn_optimizer.param_groups[0]["lr"], global_step=it)

        # if args.rank == 0 and i % args.grad_check_freq == 0:
                #print("-------------------------sanity check stn grads-------------------------------")
                #print(stn.module.transform_net.fc_localization_view1.linear2.weight.grad)
                #print(stn.module.transform_net.fc_localization_view2.linear2.weight.grad)
                #print(stn.module.transform_net.localization_net.conv2d_2.weight.grad)
                #print(f"CUDA MAX MEM:           {torch.cuda.max_memory_allocated()}")
                #print(f"CUDA MEM ALLOCATED:     {torch.cuda.memory_allocated()}")

        #if args.use_stn_optimizer and not use_pretrained_stn:
        #   stn_optimizer.zero_grad()

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
