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
1. Per object class, run few shot setting with multiple N-shots.
2. Shows under fewer data regime, what results in better plots.
3. Take an average as you did it before.

Expectation: 
We compute ECE and Generalization performance. Compute per object class.
Compare against deterministic, LA, MC-dropout, and deep ensemble.

TODO: we should get the same accuracy in all the metrics; otherwise, it doesnt make sense?
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
from clever.metric import evaluate_fewshot
from clever.metric import evaluate_query


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
    n_shot = 32 #[4] #, 8, 16, 32, 64, 128] # increase this number later.

    # load data loader
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((280, 280)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])
    train_dataset = ArmarDataset(root=args.pooldata_dir,
                                 split='train',
                                 transform=transform,
                                 sequence=0, # how about this?
                                 category=args.object)
    test_dataset = ArmarDataset(root=args.pooldata_dir,
                                split='validation',
                                transform=transform,
                                category=args.object)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, **kwargs)
    
    # load the model 
    if args.model=="TheCleverSystem":
        clever = TheCleverSystem(args=args, class_mapping={args.object: 0}, device=device) # class_mapping={"others": 0}, device=device)
    elif args.model=="TheMostCleverSystem":
        clever = TheMostCleverSystem(args=args, class_mapping={args.object: 0}, device=device)
    else:
        raise AttributeError

    # create a data loader with a subset
    N = len(train_dataset)
    nonzero = random.sample(range(0, N), int(n_shot))
    nonzero.sort()
    trainsubset = torch.utils.data.Subset(train_dataset, nonzero)
    train_loader = torch.utils.data.DataLoader(
        trainsubset, batch_size=4, shuffle=True, **kwargs)

    # train the clever
    # for i in range(0, args.num_epoch):
    clever.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, None))

    metric = evaluate_fewshot(args, clever.classifier, test_loader, device=device)
    print("metric: ", metric)

    # now evaluate the model for known and unknown objects
    sequences = ['seq0', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8']
    objects = ["apple", "bowl"]#, "cleaning_cloth", "glass_bottle", "kendo_head_cover", "keyboard", "knife", "milk", "pear", "spoon", "t_shirt"]
    known_knowns = list()
    known_unknowns = list()
    for seq_num in range(len(sequences)):
        for object_category in objects:
            # load the dataset for each objects
            test_dataset = ArmarDataset(root=args.pooldata_dir,
                                        split='validation',
                                        transform=transform,
                                        sequence=seq_num,
                                        category=object_category)
            test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=16, shuffle=False, **kwargs)

            # evaluate if query step works
            mean_prob = evaluate_query(args=args, model=clever.classifier, dataloaders=test_loader, device=device)
            print("should we query for", object_category, "?", mean_prob)
            print("Known knowns:", args.object)
            print("Known unknowns", object_category)

            # store the results
            # NOTE: we evaluate whether the system knows the knowns
            # and if the system knows the unknowns
            if object_category==args.object:
                if mean_prob < 0.75:
                    known_knowns.append(0)
                else:
                    known_knowns.append(1)
            else:
                if mean_prob < 0.75:
                    known_unknowns.append(1)
                else:
                    known_unknowns.append(0)
    print(known_knowns)
    print(known_unknowns)

    # save the results
    filename = args.results_path + args.model + "streams_" + args.object \
        + "_seed_" + str(args.seed) + "_num_epoch_" + str(args.num_epoch) + "_query_rates.json"
    metric = dict()
    metric["known_knowns"] = known_knowns
    metric["known_unknowns"] = known_unknowns
    with open(filename, "w") as file:
        json.dump(metric, file)
    # # save the results
    # filename = args.results_path + args.model + "_" + args.object \
    #     + "_seed_" + str(args.seed) + "_num_epoch_" + str(args.num_epoch) \
    #     + "_n_shot_" + str(n_shot) + ".json"
    # with open(filename, "w") as file:
    #     json.dump(metric, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("We examine how many objects our architecture can handle")
    parser.add_argument("--seed", type=int, default=1234, help="subsampling")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--results_path", help="Path to the results", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--model", type=str, default="TheMostCleverSystem", help="acquisition function")
    parser.add_argument("--object", type=str, default="apple", help="name of the object")
    parser.add_argument("--num_epoch", type=int, default=3, help="number of epochs") # critical variable
    args = parser.parse_args()
    main(args)