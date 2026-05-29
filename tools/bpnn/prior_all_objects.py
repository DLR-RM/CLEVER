#!/usr/bin/env python
# -*- coding: utf-8 -*-

################################################################################
# Copyright (c) 2024. Jongseok Lee                                             #
# All rights reserved.                                                         #
# See the accompanying LICENSE file for terms.                                 #
#                                                                              #
# Date: 01-01-2024                                                             #
# Author: Jongseok Lee                                                         #
# E-mail: jongseok.lee@dlr.de                                                  #
# Website: https://rmc.dlr.de/rm/en/staff/jongseok.lee/                        #
################################################################################

"""
Learning prior from synthetic dataset.
"""

import argparse
import torch
import os

from torchvision import transforms
from clever.data.armar import ArmarDataset
from clever.data.hows import get_pnn_dataloaders
from clever.model import clever_pnn, TheCleverNetwork
from clever.priors import init_clever_prior

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"


def main(args):
    # set the device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    objects = ['apple', 'bowl', 'cleaning_cloth', 'glass_bottle', 'kendo_head_cover', 
               'keyboard', 'knife', 'milk', 'pear', 'spoon', 't_shirt']

    # loading data-loaders
    dataloaders = []
    for an_object in objects:
        # get the data loaders. 
        use_cuda = torch.cuda.is_available()
        kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
        output_dim=2
        transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((280, 280)),
                                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
        train_dataset = ArmarDataset(root=args.root,
                                    split='train',
                                    transform=transform,
                                    category=an_object)
        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
        test_dataset = ArmarDataset(root=args.root,
                                    split='validation',
                                    transform=transform,
                                    category=an_object)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
        dataloaders.append((output_dim, (train_loader, train_loader, test_loader), torch.nn.CrossEntropyLoss()))

    if args.mode == 'train':
        # initialize and load the model
        args.model_pt_file = os.path.join(args.model_path, f'./clever_v1/full_state_dict_all_objects.pt')
        pnn = clever_pnn(args, device)

        # get the prior
        priors = init_clever_prior(model=pnn, dataloaders=dataloaders, weight_decay=1e-5)

        # declare the model
        bpnn = TheCleverNetwork(model=pnn, priors=priors, backbone=pnn.backbone, weight_decay=1e-5)

        # save the model
        torch.save(bpnn.full_state_dict(), os.path.join(args.model_path, f'./clever_v1/full_state_dict_bpnn_all_objects.pt'))
    else:
        print("Loading the model")

        # initialize the model class without loading checkpoints
        pnn = clever_pnn(args, device)

        # declare the model
        bpnn = TheCleverNetwork(model=pnn, priors=None, backbone=pnn.backbone, weight_decay=1e-5)
        bpnn.load_full_state_dict(torch.load(os.path.join(args.model_path, './clever_v1/full_state_dict_bpnn_all_objects.pt')))


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser("Loads images and labels of HOWS for a given sequence")
    parser.add_argument("--root", help="Path to the dataset location", required=True)
    parser.add_argument("--batch_size", help="Size of the batch", type=int, default=4)
    parser.add_argument("--mode", help="Mode of the dataset (train, load)", required=True)
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    args = parser.parse_args()

    main(args)