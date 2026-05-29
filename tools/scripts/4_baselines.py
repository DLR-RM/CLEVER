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
Implentation of baselines
1. MC-dropout
2. Deep ensemble
3. LA (without prior)
4. LA (without pac-bayes)
"""

import argparse
import copy
import tqdm
import random
import json
import torch
import numpy as np

from torchvision import transforms
from clever.data.armar import ArmarDataset
from clever.clever import TheMostCleverSystem
from clever.baselines import Dropout, DeepEnsemble, LaplaceApproximation, CleverWithoutPacBayes
from clever.metric import evaluate_fewshot


def main(args):
    # the seeds for controlled experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # set the device if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    use_cuda = torch.cuda.is_available()
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # initializations
    n_shots = [4, 6, 8, 10, 12, 14, 16, 18, 20]#, 32, 64, 128]

    # load data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((280, 280)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = ArmarDataset(root=args.pooldata_dir,
                                 split='train',
                                 transform=transform,
                                 category=args.object)
    test_dataset = ArmarDataset(root=args.pooldata_dir,
                                split='validation',
                                transform=transform,
                                category=args.object)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, **kwargs)

    # loop through the n-shots
    for n_shot in n_shots:
        # load the model 
        if args.model=="Dropout": # TODO: baselines here!
            baseline = Dropout(args=args, class_mapping={"other": 0}, device=device)
        elif args.model=="DeepEnsemble":
            baseline = DeepEnsemble(args=args, class_mapping={"other": 0}, device=device)
        elif args.model=="LaplaceApproximation":
            baseline = LaplaceApproximation(args=args, class_mapping={args.object: 0}, device=device)
        elif args.model=="CleverWithoutPacBayes":
            baseline = CleverWithoutPacBayes(args=args, class_mapping={args.object: 0}, device=device)
        else:
            raise AttributeError

        # create a data loader with a subset
        N = len(train_dataset)
        nonzero = random.sample(range(0, N), int(n_shot))
        nonzero.sort()
        trainsubset = torch.utils.data.Subset(train_dataset, nonzero)
        train_loader = torch.utils.data.DataLoader(
            trainsubset, batch_size=4, shuffle=True, **kwargs)
        
        # learn the model
        while True:
            baseline.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, None))
            print(baseline.metric)
            if baseline.metric['train'][0]['accuracy'] > 95.0:
                break
        # # train the baseline
        # for i in range(0, args.num_epoch):
        #     baseline.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, test_loader))

        # evaluate with test datum 
        metric = evaluate_fewshot(args, baseline.classifier, test_loader, device=device)
        print(metric)

        # save the results
        filename = args.results_path + args.model + "_" + args.object \
            + "_seed_" + str(args.seed) + "_num_epoch_" + str(args.num_epoch) \
            + "_n_shot_" + str(n_shot) + ".json"
        with open(filename, "w") as file:
            json.dump(metric, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("We examine how many objects our architecture can handle")
    parser.add_argument("--seed", type=int, default=1234, help="subsampling")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--results_path", help="Path to the results", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--model", type=str, default="Dropout", help="acquisition function")
    parser.add_argument("--object", type=str, default="apple", help="name of the object")
    parser.add_argument("--num_epoch", type=int, default=3, help="number of epochs") # critical variable
    args = parser.parse_args()
    main(args)