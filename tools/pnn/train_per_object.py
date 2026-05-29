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
1. Training PNN on synthetic dataset with selected classes.
"""

import argparse
import torch
import torch.nn.functional as F
import os

from torchvision import transforms
from torch import nn
from bpnn.pnn import dataset_step_pnn

from clever.data.armar import ArmarDataset
from clever.data.hows import get_pnn_data_tuple
from clever.trainer import fit_pnn_with_hows, evaluate_pnn_on_hows, pnn_fit_the_unknowns
from clever.model import clever_pnn

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"


def main(args):
    # set the device and seed
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initialize the model
    pnn = clever_pnn(args, device)

    # load the data loaders
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((280, 280)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = ArmarDataset(root=args.root,
                                 split='train',
                                 transform=transform,
                                 category=args.object)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, **kwargs)
    test_dataset = ArmarDataset(root=args.root,
                                split='validation',
                                transform=transform,
                                category=args.object)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, **kwargs)
    dataloaders = (train_loader, train_loader, test_loader)

    # starting training the pnn
    if args.mode == "train":
        metric, classifier = pnn_fit_the_unknowns(model=pnn,
                                                  dataloader=dataloaders,
                                                  device=device,
                                                  use_validation_set=True,
                                                  num_epochs=1,
                                                  is_classification=[True for i in range(1)])
        torch.save(
            classifier.full_state_dict(), 
            os.path.join(args.model_path+'/clever_v1/', f'full_state_dict_{args.object}.pt')
            )
        print("training metrics", metric)
    else:
        print("Wrong args.mode input")


if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser("Loads images and labels for a given sequence")
    parser.add_argument("--root", help="Path to the dataset location", required=True)
    parser.add_argument("--batch_size", help="Size of the batch", type=int, default=4)
    parser.add_argument("--mode", help="Mode of the dataset (train, validation)", required=True)
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--object", type=str, default="apple", help="name of the object")

    args = parser.parse_args()

    main(args)
    