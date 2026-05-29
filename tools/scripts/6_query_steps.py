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
Run the entire set up by having them run as sequences.

Results on (1) query success rate, (2) number of query steps.

TheCleverSystem and TheMostCleverSystem
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
from clever.metric import evaluate_uncertainty_reduction


# TODO: just save the right values, which is accuracy.
# TODO: think about waht to do with ...

def main(args):
    # the seeds for controlled experiments
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)

    # active learning hyperparameters - initializations
    test_batch_size = 16
    batch_size = 16
    scoring_batch_size = 8
    use_cuda = torch.cuda.is_available()
    device = "cuda" if use_cuda else "cpu"
    kwargs = {"num_workers": 1, "pin_memory": True} if use_cuda else {}

    # initializations
    sequences = ['seq0', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8']
    knowns = list()
    unknowns = list()

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
    initial_samples = get_balanced_sample_indices(
        get_targets(train_dataset), num_classes=args.num_classes, n_per_digit=args.num_initial_samples/args.num_classes
    )

    # dataloader functionalities - split into test, pool and training.
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=test_batch_size, shuffle=False, **kwargs)
    active_learning_data = ActiveLearningData(train_dataset)
    active_learning_data.acquire(initial_samples)
    train_loader = torch.utils.data.DataLoader(
        active_learning_data.training_dataset,
        batch_size=batch_size,
        shuffle=True,
        **kwargs,
    )

    # load the model
    if args.model=="TheCleverSystem":
        clever = TheCleverSystem(args=args, class_mapping={"others": 0}, device=device)
    elif args.model=="TheMostCleverSystem":
        clever = TheMostCleverSystem(args=args, class_mapping={args.object: 0}, device=device)
    else:
        raise AttributeError

    # starting the active learning experiments
    test_accs, test_loss, added_indices = [], [], []

    for seq_num in range(len(sequences)):
        # Acquire pool predictions
        N = len(active_learning_data.pool_dataset)
        if args.subsample:
            nonzero = random.sample(range(0, N), args.num_subsample)
            nonzero.sort()
            poolsubset = torch.utils.data.Subset(active_learning_data.pool_dataset, nonzero)
            pool_loader = torch.utils.data.DataLoader(
                poolsubset, batch_size=scoring_batch_size, shuffle=False, **kwargs)
            N = args.num_subsample
            logits_N_K_C = torch.empty((N, args.num_inference_samples, args.num_classes), dtype=torch.double, pin_memory=use_cuda)
        else:
            pool_loader = torch.utils.data.DataLoader(
                active_learning_data.pool_dataset, batch_size=scoring_batch_size, shuffle=False, **kwargs)
            logits_N_K_C = torch.empty((N, args.num_inference_samples, args.num_classes), dtype=torch.double, pin_memory=use_cuda)

        if args.acquisition=="batchbald" or args.acquisition=="bald":
            with torch.no_grad():
                clever.classifier.eval()
                for i, (inputs, _) in enumerate(pool_loader):
                    inputs = inputs.to(device=device)
                    lower = i * pool_loader.batch_size
                    upper = min(lower + pool_loader.batch_size, N)

                    if args.model=="TheCleverSystem":
                        outputs = clever.classifier(inputs)
                        outputs = [torch.stack([output for i in range(0, args.num_inference_samples)], dim=1) for output in outputs]
                    elif args.model=="TheMostCleverSystem":
                        outputs = clever.classifier.bayesian_analysis(inputs, num_samples=30)
                        outputs = [torch.transpose(output, 1, 2) for output in outputs]
                    else:
                        raise AttributeError

                    logits_N_K_C[lower:upper].copy_(outputs[clever.class_mapping[args.object]].double(), non_blocking=True)

        with torch.no_grad():
            if args.acquisition=="batchbald":
                candidate_batch = get_batchbald_batch(
                    logits_N_K_C, args.acquisition_batch_size, args.num_samples, dtype=torch.double, device=device
                ) # TODO: understand the influence of num_samples
            elif args.acquisition=="bald":
                candidate_batch = get_bald_batch(
                    logits_N_K_C, args.acquisition_batch_size, dtype=torch.double, device=device
                )
            elif args.acquisition=="random":
                candidate_batch = get_random_batch(
                    logits_N_K_C, args.acquisition_batch_size, dtype=torch.double, device=device
                )
            else:
                raise AttributeError
                
        if args.subsample: 
            candidate_batch.indices = np.asarray(nonzero)[candidate_batch.indices]

        targets = get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)

        print("Dataset indices: ", dataset_indices)
        print("Scores: ", candidate_batch.scores)
        print("Labels: ", targets[candidate_batch.indices])

        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(len(dataset_indices))

        # train with the object categorical class 
        clever.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, test_loader)) 
        print("Accuracy of the deterministic model: ", clever.metric['test'])
        test_loss.append(clever.metric['test'][0]['loss'])
        test_accs.append(clever.metric['test'][0]['accuracy'])

        # reset the pool set
        train_dataset = ArmarDataset(root=args.pooldata_dir,
                                    split='train',
                                    transform=transform,
                                    sequence=seq_num,
                                    category=args.object)
        initial_samples = get_balanced_sample_indices(
            get_targets(train_dataset), num_classes=args.num_classes, n_per_digit=args.num_initial_samples/args.num_classes
        )
        active_learning_data = ActiveLearningData(train_dataset)
        active_learning_data.acquire(initial_samples)

        train_loader = torch.utils.data.DataLoader(
            active_learning_data.training_dataset,
            batch_size=batch_size,
            shuffle=True,
            **kwargs,
        )
        del pool_loader
    
    # saving the results
    results = {}
    results['test_loss'] = test_loss
    results['test_accs'] = test_accs

    # save the latest one
    filename = args.results_path + "6_query_steps_" + args.acquisition + "_" + args.model + "_" + args.object \
        + "_subsample_" + str(args.subsample) + "_seed_" + str(args.seed) + "_num_initial_samples_" + \
            str(args.num_initial_samples) + "_max_training_samples_" + str(args.max_training_samples) + \
               "_acquisition_batch_size_" + str(args.acquisition_batch_size) + "_num_subsample_" + str(args.num_subsample) + ".json"
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
    parser.add_argument("--model", type=str, default="TheCleverSystem", help="acquisition function")
    parser.add_argument("--object", type=str, default="apple", help="name of the object")
    parser.add_argument("--subsample", type=bool, default=False, help="subsampling")
    parser.add_argument("--seed", type=int, default=1234, help="subsampling")
    parser.add_argument("--num_epoch", type=int, default=1, help="number of epochs")

    parser.add_argument("--num_initial_samples", type=int, default=20, help="initial sample size")
    parser.add_argument("--num_classes", type=int, default=2, help="number of classes")
    parser.add_argument("--num_inference_samples", type=int, default=30, help="MC samples")
    parser.add_argument("--num_samples", type=int, default=100000, help="samples for batchbald approximation")
    parser.add_argument("--max_training_samples", type=int, default=150, help="maximum training set")
    parser.add_argument("--acquisition_batch_size", type=int, default=5, help="acquisition batch size")
    parser.add_argument("--num_subsample", type=int, default=40, help="ratio of the subsampling")

    args = parser.parse_args()
    main(args)