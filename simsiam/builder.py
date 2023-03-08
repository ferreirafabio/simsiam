# Copyright (c) Facebook, Inc. and its affiliates.
# All rights reserved.

# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.

import torch
import torch.nn as nn


class SimSiam(nn.Module):
    """
    Build a SimSiam model.
    """
    def __init__(self, base_encoder, dim=2048, pred_dim=512, theta_layer_dim=None):
        """
        dim: feature dimension (default: 2048)
        pred_dim: hidden dimension of the predictor (default: 512)
        """
        super(SimSiam, self).__init__()

        self.theta_layer_dim = theta_layer_dim
        # create the encoder
        # num_classes is the output fc dimension, zero-initialize last BNs
        self.encoder = base_encoder(num_classes=dim, zero_init_residual=True)

        # build a 3-layer projector
        prev_dim = self.encoder.fc.weight.shape[1]
        self.encoder.fc = nn.Sequential(nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # first layer
                                        nn.Linear(prev_dim, prev_dim, bias=False),
                                        nn.BatchNorm1d(prev_dim),
                                        nn.ReLU(inplace=True), # second layer
                                        self.encoder.fc,
                                        nn.BatchNorm1d(dim, affine=False)) # output layer
        self.encoder.fc[6].bias.requires_grad = False # hack: not use bias as it is followed by BN

        # build a 2-layer predictor
        self.predictor = nn.Sequential(nn.Linear(dim, pred_dim, bias=False),
                                        nn.BatchNorm1d(pred_dim),
                                        nn.ReLU(inplace=True), # hidden layer
                                        nn.Linear(pred_dim, dim)) # output layer
        
        self.view_reconstructor = None
        if self.theta_layer_dim:

            # self.view_reconstructor = nn.Sequential(nn.Linear(self.theta_layer_dim+dim, 2048),
            #                                         nn.BatchNorm1d(2048),
            #                                         nn.ReLU(inplace=True),
            #                                         nn.Unflatten(1, (32, 8, 8)),
            #                                         # ((8-1)*2)-(2*2)+(6-1)+1
            #                                         nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=5, stride=2, padding=2), # 8x8 -> 15x15
            #                                         nn.BatchNorm2d(16),
            #                                         nn.ReLU(inplace=True),
            #                                         # (15-1)*2-(2*1)+(6-1)+1
            #                                         nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=6, stride=2, padding=1), # 15x15 -> 32x32
            #                                         )

            self.view_reconstructor = nn.Sequential(nn.Linear(self.theta_layer_dim+dim, 2048),
                                                    nn.LayerNorm(2048),
                                                    nn.LeakyReLU(inplace=True),
                                                    nn.Unflatten(1, (32, 8, 8)),
                                                    # ((8-1)*1)-(2*1)+(3-1)+1
                                                    nn.ConvTranspose2d(in_channels=32, out_channels=32, kernel_size=3, stride=1, padding=1), # 8x8 -> 8x8
                                                    nn.BatchNorm2d(32),
                                                    nn.LeakyReLU(inplace=True),
                                                    # ((8-1)*2)-(2*1)+(3-1)+1
                                                    nn.ConvTranspose2d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1), # 8x8 -> 15x15
                                                    nn.BatchNorm2d(16),
                                                    nn.LeakyReLU(inplace=True),
                                                    # ((15-1)*2)-(2*1)+(4-1)+1
                                                    nn.ConvTranspose2d(in_channels=16, out_channels=16, kernel_size=4, stride=2, padding=1), # 15x15 -> 30x30
                                                    nn.BatchNorm2d(16),
                                                    nn.LeakyReLU(inplace=True),
                                                    # ((30-1)*1)-(2*1)+(5-1)+1
                                                    nn.ConvTranspose2d(in_channels=16, out_channels=3, kernel_size=5, stride=1, padding=1), # 29x29 -> 32x32
            )


    def forward(self, x, theta):
        """
        Input:
            x1: first views of images
            x2: second views of images
        Output:
            p1, p2, z1, z2: predictors and targets of the network
            See Sec. 3 of https://arxiv.org/abs/2011.10566 for detailed notations
        """

        # compute features for one view
        z = self.encoder(x) # NxC
        p = self.predictor(z) # NxC
        # print(p.shape)
        # print(theta.shape)
        input_tensor = torch.cat([p, theta], dim=1)
        view_recon = self.view_reconstructor(input_tensor)

        return view_recon

      
