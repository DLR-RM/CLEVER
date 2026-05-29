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
Measuring ECE, Query rates, Query numbers for the classification tasks.
"""

# TODO:
# 1. template for programming.
# 2. changing the data loaders.
# 3. changing the prediction models/clever.
# 4. logging and running.

import argparse
import copy
import tqdm
import random
import json
import torch
import numpy as np

from torchvision import transforms
from bpnn.pnn import ProgressiveNeuralNetwork
from clever.data.armar import ArmarDataset, ArmarSequential
from clever.activelearner import get_bald_batch, get_batchbald_batch, get_random_batch, \
    get_balanced_sample_indices, ActiveLearningData, get_targets, CandidateBatch
from clever.trainer import pnn_fit_the_knowns, pnn_fit_the_unknowns, \
    bpnn_fit_the_unknowns, bpnn_fit_the_knowns
from clever.model import ClassifierNet
from clever.clever import TheMostCleverSystem, TheCleverSystem
from clever.metric import evaluate_query, evaluate_fewshot

# TODO: test
# TODO: include active learning sample
# TODO: extend to full sequence and go home.


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
    transform = transforms.Compose([transforms.ToTensor(), transforms.Resize((280, 280)),
                                    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])])

    # load the model
    if args.model=="TheCleverSystem":
        clever = TheCleverSystem(args=args, class_mapping={"others": 0}, device=device)
        clever.classifier.add_new_column(is_classification=True, output_size=2)
        clever.class_mapping = {"others": 0, args.object: 1}
    elif args.model=="TheMostCleverSystem":
        clever = TheMostCleverSystem(args=args, class_mapping={args.object: 0}, device=device)
    else:
        raise AttributeError

    # if the_most_clever; then train the model beforehand with synthetic set
    if args.model=="TheMostCleverSystem":
        sim_dataset = ArmarDataset(root=args.pooldata_dir,
                                    split='train',
                                    transform=transform,
                                    sequence=0,
                                    category=args.object) # TODO: it should first say 0 and 1
        sim_loader = torch.utils.data.DataLoader(sim_dataset, batch_size=16, shuffle=True, **kwargs)
        val_dataset = ArmarDataset(root=args.pooldata_dir,
                                    split='validation',
                                    transform=transform,
                                    sequence=0,
                                    category=args.object) # TODO: it should first say 0 and 1
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=16, shuffle=False, **kwargs)
        clever.active_learn(object_name=args.object, dataloader=(sim_loader, val_loader, None))
    
    # starting the active learning experiments
    sequences = ['seq0', 'seq1', 'seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq7', 'seq8', 'seq9']
    objects = ["apple", "bowl", "cleaning_cloth", "glass_bottle", \
        "kendo_head_cover", "keyboard", "knife", "milk", "pear", "spoon", "t_shirt"]
    test_accs, test_ece, test_success_rates, test_probs = [], [], [], []
    seq_list = []

    for sequence in range(0, len(sequences)):
        seq_list.append(sequence)
        added_indices = []

        # evaluate query decision
        for object_category in objects:
            print("model length", len(clever.classifier.networks))
            print("current object category:", object_category)
            query_dataset = ArmarDataset(root=args.pooldata_dir,
                                         split='validation',
                                         transform=transform,
                                         sequence=sequence,
                                         category=object_category) # TODO: it should first say 0 and 1
            query_loader = torch.utils.data.DataLoader(query_dataset, batch_size=4, shuffle=False, **kwargs)
            mean_prob = evaluate_query(args=args, model=clever.classifier, dataloaders=query_loader, device=device)
            test_probs.append(mean_prob)
            print("mean probabilities:", mean_prob)
            if object_category==args.object:
                if mean_prob < 0.75:
                    test_success_rates.append(0)
                else:
                    test_success_rates.append(1)
            else:
                if mean_prob < 0.75:
                    test_success_rates.append(1)
                else:
                    test_success_rates.append(0)
        print("currnet query rates", test_success_rates)
        print("test_probs", test_probs)

        # active selection of samples (init to soemthing)
        train_dataset = ArmarSequential(root=args.pooldata_dir,
                                        split='train',
                                        transform=transform,
                                        sequence=seq_list,
                                        balancer=objects,
                                        object_names=objects,
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
        print("len(active_learning_data.training_dataset)", len(active_learning_data.training_dataset))

        # Acquire pool predictions
        N = len(active_learning_data.pool_dataset)
        nonzero = random.sample(range(0, N), args.num_subsample)
        nonzero.sort()
        poolsubset = torch.utils.data.Subset(active_learning_data.pool_dataset, nonzero)
        pool_loader = torch.utils.data.DataLoader(
            poolsubset, batch_size=scoring_batch_size, shuffle=False, **kwargs)
        N = args.num_subsample
        logits_N_K_C = torch.empty((N, args.num_inference_samples, args.num_classes), dtype=torch.double, pin_memory=use_cuda)

        if args.acquisition=="batchbald":
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
                        outputs = clever.classifier.bayesian_analysis(inputs, num_samples=args.num_inference_samples)
                        outputs = [torch.transpose(output, 1, 2) for output in outputs]
                    else:
                        raise AttributeError
                    logits_N_K_C[lower:upper].copy_(outputs[clever.class_mapping[args.object]].double(), non_blocking=True)

        with torch.no_grad():
            if args.acquisition=="batchbald":
                candidate_batch = get_batchbald_batch(
                    logits_N_K_C, args.acquisition_batch_size, args.num_samples, dtype=torch.double, device=device
                )
            elif args.acquisition=="random":
                candidate_batch = get_random_batch(
                    logits_N_K_C, args.acquisition_batch_size, dtype=torch.double, device=device
                )
            else:
                raise AttributeError
        candidate_batch.indices = np.asarray(nonzero)[candidate_batch.indices]
        targets = get_targets(active_learning_data.pool_dataset)
        dataset_indices = active_learning_data.get_dataset_indices(candidate_batch.indices)
        active_learning_data.acquire(candidate_batch.indices)
        added_indices.append(len(dataset_indices))
        print("len(active_learning_data.training_dataset)", len(active_learning_data.training_dataset))

        # evaluate accuracy and ece
        test_dataset = ArmarDataset(root=args.pooldata_dir,
                                    split='validation',
                                    transform=transform,
                                    sequence=sequence,
                                    category=args.object)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=4, shuffle=False, **kwargs)
        
        # learn the model
        while True:
            clever.active_learn(object_name=args.object, dataloader=(train_loader, train_loader, None))
            print("clever.metric", clever.metric)
            if clever.metric['train'][0]['accuracy'] > 95.0:
                print(clever.metric['train'][0]['accuracy'])
                print("breaking up")
                break

        metric = evaluate_fewshot(args, clever.classifier, test_loader, device=device)
        test_accs.append(metric["test_prec@1"])
        test_ece.append(metric["test_ece"])
        print("metric", metric)
        print("metric: ", metric["test_prec@1"], metric["test_ece"])

        del pool_loader

    # saving the results
    results = {}
    results['test_probs'] = test_probs
    results['test_ece'] = test_ece
    results['test_accs'] = test_accs
    results['test_success_rates'] = test_success_rates

    # save the latest one
    filename = args.results_path + "5_classification_" + args.model + "_" + args.object \
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
    parser.add_argument("--object", type=str, help="name of the object")
    parser.add_argument("--subsample", type=bool, default=True, help="subsampling")
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
