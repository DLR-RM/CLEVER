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

import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle
import robustness_metrics as rm
import torch
import torch.nn.functional as F

from scipy.ndimage import binary_dilation
from PIL import Image
from typing import Optional, Callable, Tuple, Any, Dict, List

from clever.predictions import cartesian, barycentric


def evaluate(probs, labels_test):
    """Evaluates metrics.
    Both inputs in numpy arrays. Prob in Batch x Class where Target is Batch
    """
    # Accuracy.
    int_preds = np.argmax(probs, axis=1)
    accuracy = np.mean(int_preds == labels_test)

    # ECE
    ece = rm.metrics.ExpectedCalibrationError(num_bins=15)
    ece.add_batch(probs, label=labels_test)
    ece_res = ece.result()["ece"]

    # NLL
    nll_metric = rm.metrics.NegativeLogLikelihood()
    nll_metric.add_batch(probs, label=labels_test)
    nll_res = nll_metric.result()["nll"]

    # AUC
    calib_auc = rm.metrics.CalibrationAUC(correct_pred_as_pos_label=False)
    int_preds = np.argmax(probs, axis=-1)
    confidence = np.max(probs, axis=-1)
    calib_auc.add_batch(
    int_preds, label=labels_test, confidence=confidence.astype("float32"))
    calib_auc_res = calib_auc.result()["calibration_auc"]

    metric_results = {
    "test_prec@1": accuracy,
    "test_ece": ece_res,
    "test_nll": nll_res,
    "test_calib_auc": calib_auc_res,
    }
    return metric_results

def metric_saver(filename, data):
    with open(filename, 'wb') as handle:
        pickle.dump(data, handle, protocol=pickle.HIGHEST_PROTOCOL)

def metric_loader(filename):
    with open(filename, 'rb') as handle:
        data = pickle.load(handle)
    return data

def evaluate_fewshot(args, model, dataloaders, device, metrics_dim=-1):
    """Evaluates ProgressiveNeuralNetworks.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: test data loader

    Returns:
        A dict containing the evaluation metrics
    """
    # model preparations
    if args.model=="DeepEnsemble":
        [classifier.eval() for classifier in model]
        [classifier.requires_grad_(False) for classifier in model]
        [classifier.to(device) for classifier in model]
    else:
        model.eval()
        model.requires_grad_(False)
        model.to(device)

    target_list = []
    prob_list = []

    with torch.no_grad():
        for iter_num, (features, targets) in enumerate(dataloaders):
            features, targets = features.to(device), targets.to(device)
            if args.model=="TheCleverSystem":
                logits = model(features)
                probs = F.softmax(logits[metrics_dim], dim=-1)
            elif args.model=="TheMostCleverSystem" or args.model=="CleverWithoutPacBayes":
                logits = model.bayesian_analysis(features, num_samples=100)
                probs = F.softmax(logits[metrics_dim].permute(0, 2, 1), dim=-1)
                probs = probs.mean(dim=1)
            elif args.model=="LaplaceApproximation":
                logits = model.bayesian_analysis(features, num_samples=100)
                probs = F.softmax(logits[metrics_dim].permute(0, 2, 1), dim=-1)
                probs = probs.mean(dim=1)
            elif args.model=="Dropout":
                logits = model(features)
                probs = F.softmax(logits[metrics_dim], dim=-1)
                probs = probs.mean(dim=1)
            elif args.model=="DeepEnsemble":
                logits = [classifier(features) for classifier in model]
                probs = [F.softmax(logit[metrics_dim], dim=-1) for logit in logits]
                probs = torch.stack(probs)
                probs = probs.mean(dim=0)
            else:
                raise NotImplementedError
            target_list.append(targets.detach().cpu().numpy())
            prob_list.append(probs.detach().cpu().numpy())

    prob_numpy = np.concatenate(prob_list, axis=0)
    target_numpy = np.concatenate(target_list, axis=0)
    test_metrics=evaluate(prob_numpy, target_numpy)
    return test_metrics

def evaluate_query(args, model, dataloaders, device, metrics_dim=-1):
    """[summary]

    Args:
        args ([type]): [description]
        model ([type]): [description]
        dataloaders ([type]): [description]
        device ([type]): [description]
        metrics_dim (int, optional): [description]. Defaults to 0.

    Raises:
        NotImplementedError: [description]
    """
    # model evaluate
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    # initialization
    target_list = []
    prob_list = []

    # predictions
    with torch.no_grad():
        for iter_num, (features, targets) in enumerate(dataloaders):
            features, targets = features.to(device), targets.to(device)
            if args.model=="TheCleverSystem":
                logits = model(features)
                probs = F.softmax(logits[metrics_dim], dim=-1)
            elif args.model=="TheMostCleverSystem": # something wrong here?
                metrics_dim=0
                logits = model.bayesian_analysis(features, num_samples=100)
                probs = F.softmax(logits[metrics_dim].permute(0, 2, 1), dim=-1)
                probs = probs.mean(dim=1)
            else:
                raise NotImplementedError
            target_list.append(targets.detach().cpu().numpy())
            prob_list.append(probs.detach().cpu().numpy())
    prob_numpy = np.concatenate(prob_list, axis=0)
    target_numpy = np.concatenate(target_list, axis=0)

    # multiply target numpy and prob and remove all zeros
    final_prob = np.multiply(prob_numpy[:,1], target_numpy)
    final_prob = final_prob[final_prob != 0]
    return np.mean(final_prob)

def evaluate_uncertainty_reduction(args, model, dataloaders, device, metrics_dim=0):
    """[summary]

    Args:
        args ([type]): [description]
        model ([type]): [description]
        dataloaders ([type]): [description]
        device ([type]): [description]
        metrics_dim (int, optional): [description]. Defaults to 0.

    Raises:
        NotImplementedError: [description]
    """
    # model evaluate
    model.eval()
    model.requires_grad_(False)
    model.to(device)

    # initialization
    target_list = []
    prob_list = []

    # predictions
    with torch.no_grad():
        for iter_num, (features, targets) in enumerate(dataloaders):
            features, targets = features.to(device), targets.to(device)
            if args.model=="TheCleverSystem":
                logits = model(features)
                probs = F.softmax(logits[metrics_dim], dim=-1)
            elif args.model=="TheMostCleverSystem":
                logits = model.bayesian_analysis(features, num_samples=100)
                probs = F.softmax(logits[metrics_dim].permute(0, 2, 1), dim=-1)
                probs = probs.mean(dim=1)
            else:
                raise NotImplementedError
            target_list.append(targets.detach().cpu().numpy())
            prob_list.append(probs.detach().cpu().numpy())
    prob_numpy = np.concatenate(prob_list, axis=0)
    target_numpy = np.concatenate(target_list, axis=0)

    # multiply target numpy and prob and remove all zeros
    final_prob_knowns = np.multiply(prob_numpy[:,1], target_numpy)
    final_prob_knowns = final_prob_knowns[final_prob_knowns != 0]
    where_0 = np.where(target_numpy == 0)
    where_1 = np.where(target_numpy == 1)
    target_numpy[where_0] = 1
    target_numpy[where_1] = 0
    final_prob_unknowns = np.multiply(prob_numpy[:,1], target_numpy)
    final_prob_unknowns = final_prob_unknowns[final_prob_unknowns != 0]
    return final_prob_knowns, final_prob_unknowns