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
Training without active learning: the entire dataset.
"""
import argparse
import copy
import tqdm
import random
import json
import torch
import numpy as np

from torchvision import transforms
from bpnn.pnn import ProgressiveNeuralNetwork
from clever.data.armar import ArmarDataset
from clever.activelearner import get_bald_batch, get_batchbald_batch, get_random_batch, \
    get_balanced_sample_indices, ActiveLearningData, get_targets, CandidateBatch
from clever.trainer import pnn_fit_the_knowns, pnn_fit_the_unknowns, \
    bpnn_fit_the_unknowns, bpnn_fit_the_knowns
from clever.model import ClassifierNet
from clever.clever import TheMostCleverSystem, TheCleverSystem


def main(args):
    # the seeds for controlled experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # active learning hyperparameters - initializations
    test_batch_size = 16
    batch_size = 16
    scoring_batch_size = 2
    column_num = 1
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # data set definitions
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

    # dataloader functionalities - split into test, pool and training.
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, **kwargs)

    # load the model
    if args.model=="TheCleverSystem":
        clever = TheCleverSystem(args=args, device=device)
    elif args.model=="TheMostCleverSystem":
        clever = TheMostCleverSystem(args=args, device=device)
    else:
        raise AttributeError

    # starting the active learning experiments
    test_accs, test_loss, train_loss, train_accs = [], [], [], []

    for i in range(0, args.num_epoch):
        # train with the object categorical class 
        clever.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, test_loader)) #test_loader))
        train_loss.append(clever.metric['train'][0]['loss'])
        train_accs.append(clever.metric['train'][0]['accuracy'])
        test_loss.append(clever.metric['test'][0]['loss'])
        test_accs.append(clever.metric['test'][0]['accuracy'])

    # saving the results
    results = {}
    results['num_initial_samples'] = args.num_initial_samples
    results['num_classes'] = args.num_classes
    results['num_inference_samples'] = args.num_inference_samples
    results['num_samples'] = args.num_samples 
    results['max_training_samples'] = args.max_training_samples 
    results['acquisition_batch_size'] = args.acquisition_batch_size
    results['subsample_ratio'] = args.subsample_ratio
    results['test_loss'] = test_loss
    results['test_accs'] = test_accs

    # save the latest one
    filename = args.results_path + "2_model_" + args.acquisition + "_" + args.model + "_" + args.object \
        + "_subsample_" + str(args.subsample) + "_seed_" + str(args.seed) + "_num_initial_samples_" + \
            str(args.num_initial_samples) + "_max_training_samples_" + str(args.max_training_samples) + \
               "_acquisition_batch_size_" + str(args.acquisition_batch_size) + "_subsample_ratio_" + str(args.subsample_ratio) + ".json"
    with open(filename, "w") as file:
        json.dump(results, file)

    return None

if __name__ == "__main__":
    parser = argparse.ArgumentParser("We examine how many objects our architecture can handle")

    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")
    
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--results_path", help="Path to the results", required=True)
    parser.add_argument("--acquisition", type=str, default="bald", help="acquisition function")
    parser.add_argument("--model", type=str, default="TheCleverSystem", help="model name")
    parser.add_argument("--object", type=str, default="apple", help="name of the object")
    parser.add_argument("--subsample", type=bool, default=True, help="subsampling")
    parser.add_argument("--seed", type=int, default=1234, help="seed")
    parser.add_argument("--num_epoch", type=int, default=1, help="number of epochs")

    parser.add_argument("--num_initial_samples", type=int, default=20, help="initial sample size")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--num_inference_samples", type=int, default=30, help="MC samples")
    parser.add_argument("--num_samples", type=int, default=100000, help="samples for batchbald approximation")
    parser.add_argument("--max_training_samples", type=int, default=150, help="maximum training set")
    parser.add_argument("--acquisition_batch_size", type=int, default=5, help="acquisition batch size")
    parser.add_argument("--subsample_ratio", type=float, default=0.05, help="ratio of the subsampling")

    args = parser.parse_args()
    main(args)