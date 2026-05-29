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

import os
import numpy as np
import torch

from tqdm import tqdm
from typing import List, Dict, Union, Optional, Any, Tuple, Callable, Iterable
from torch import nn, Tensor
from torch.utils.data import DataLoader, Dataset

from bpnn.utils import base_path, run_epoch, get_dataset_and_name, get_verbose_string
from bpnn.pnn import ProgressiveNeuralNetwork
from bpnn.criterions import Criterion
from bpnn.curvature_scalings import CurvatureScaling, McAllesterScaling
from bpnn.utils import device, fit, compute_curvature
from curvature.curvatures import *

from clever.data.hows import get_hows_dataloaders
from clever.model import TheCleverNetwork, McAllesterCriterion


def fit_pnn_with_hows(
        model: ProgressiveNeuralNetwork,
        dataloaders: List[Tuple[Optional[int], str, nn.Module]],
        dataset_step: Callable,
        data_dir: str,
        batch_size: int = 16,
        weight_decay: float = 1e-5,
        learning_rate: float = 2e-3,
        num_epochs: int = 100,
        patience: int = 10,
        name: Optional[str] = None,
        eval_every_task: bool = True,
        **kwargs) -> List[Dict]:
    """Runs the full optimization routine for ProgressiveNeuralNetworks.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: A tuple of output dimensions, train-, val-, and test-
            dataloader names and loss function
        dataset_step: A function that represents adding and training a new column
        weight_decay: The weight decay used in the PNN
        learning_rate: The learning rate used in the full training
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        name: The name of the run (also the save path)
        eval_every_task: Whether the model should be evaluated on each column
            after training a new column

    Returns:
        A dict containing the training and evaluation metrics
    """
    metrics = []
    model_path = os.path.join(base_path, 'data/model', name) # TODO: specify the base_path
    if name is not None and not os.path.exists(model_path):
        os.makedirs(model_path)

    if name is not None:
        torch.save(model.full_state_dict(), os.path.join(model_path, 'full_state_dict_-1.pt')) # TODO: specify the name

    previous_tasks = model.previous_tasks # +1 for choosing current task
    for task, (output_size, sequence, loss_function) in enumerate(dataloaders[previous_tasks:], start=previous_tasks):
        hows_dataloader = get_hows_dataloaders(data_dir=data_dir, batch_size=batch_size, sequence=sequence)
        
        if not isinstance(loss_function, (nn.CrossEntropyLoss, nn.MSELoss, nn.BCEWithLogitsLoss)):
            raise ValueError('Only CrossEntropyLoss, BCEWithLogitsLoss and MSELoss are currently supported')
        torch.cuda.empty_cache()

        train_dataset, dataset_name = get_dataset_and_name(hows_dataloader[0])

        train_metrics = dataset_step(
            model, hows_dataloader, loss_function, output_size, weight_decay,
            learning_rate, train_dataset=train_dataset,
            num_epochs=num_epochs, patience=patience, **kwargs)

        if name is not None:
            torch.save(model.full_state_dict(), os.path.join(model_path, f'full_state_dict_{task}.pt'))

        # evaluate model
        first_task = 1 if eval_every_task else task
        is_classification = [type(loss_function) is type(nn.CrossEntropyLoss()) or type(loss_function) is type(nn.BCEWithLogitsLoss())
                             for (_, _, loss_function), _ in zip(dataloaders[1:], range(task))]
        metrics.append(
            {
                'train': train_metrics,
                'test': evaluate_pnn_on_hows(
                    model, dataloaders[first_task:task + 1],
                    range(first_task - 1, task),
                    data_dir=data_dir,
                    is_classification=is_classification)
            })
    return metrics

def evaluate_pnn_on_hows(
        model: ProgressiveNeuralNetwork,
        dataloaders: List[Tuple[Optional[int], str, nn.Module]],
        dims: Iterable[int],
        data_dir: str,
        batch_size: int = 16,
        metrics: Union[str, List[str]] = 'all',
        is_classification: Union[bool, List[bool]] = True):
    """Evaluates ProgressiveNeuralNetworks.

    Args:
        model: A ProgressiveNeuralNetwork object
        dataloaders: A tuple of output dimensions, train-, val-, and
            test-dataloaders strings and loss function
        dims: The columns that should be used for each dataloader
        metrics: 'all' or a list of metrics (see bpnn.utils.MetricsSet)
        is_classification: Whether the task is a classification task

    Returns:
        A dict containing the evaluation metrics
    """
    model.eval()
    model.requires_grad_(False)
    test_metrics = []
    for local_dim, (_, sequence, loss_function) in zip(dims, dataloaders):
        # prepare dataloader
        hows_dataloader = get_hows_dataloaders(data_dir=data_dir, batch_size=batch_size, sequence=sequence)

        # run the epoch
        test_metric = run_epoch(
            model, hows_dataloader[-1], loss_function,
            metrics=metrics, metrics_dim=local_dim,
            is_classification=is_classification)
        
        # collect metrics
        local_dataset, local_dataset_name = get_dataset_and_name(hows_dataloader[-1])
        test_metric['dataset'] = local_dataset_name
        test_metrics.append(test_metric)

    return test_metrics

def pnn_fit_the_knowns(
        model: nn.Module,
        dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader],
        device: str,
        column_num: int,
        criterion: Callable[[Tensor, Tensor], Tensor]=nn.CrossEntropyLoss(),
        weight_decay: float = 1e-5,
        is_classification: Union[bool, List[bool]] = True,
        learning_rate: float = 2e-3,
        use_validation_set: bool = True,
        num_epochs: int = 100,
        patience: int = 10,
        metrics_run_epoch: Union[str, List[str]] = None) \
        -> Dict[str, List[Dict[str, float]]]:
    """Fitting the network for the known objects by updating a specific column.

    Args:
        model: A PyTorch model
        dataloader:  A tuple of train-, val-, and test-dataloaders
        criterion: A function mapping the logits and targets to the loss
        weight_decay: The weight decay used in the optimizer
        device: Either cpu or cuda.
        column_num: n-th column of interest for updating the model.
        is_classification: A list of booleans indicating whether the metric
            is a classification metric or a regression metric (or a single
            boolean if all metrics are of the same type)
        learning_rate: The learning rate used in the Adam optimizer
        use_validation_set: Whether the training or validation set should be
            used for early stopping
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        metrics_run_epoch: Either 'all' or a list of metrics strings
            (Possible values: 'accuracy', 'brier score',
            'negative log likelihood', 'entropy', 'calibration',
            'entropy histogram')

    Returns:
        A dict containing the training, validation and test metrics
    """
    # enable training of only a column  
    model.base_network.requires_grad_(False)
    model.networks.requires_grad_(False)
    model.networks.eval()

    intra_column = [column for column in model.networks]
    intra_column[column_num].requires_grad_(True)
    intra_column[column_num].train()

    # metrics and dataloading
    if metrics_run_epoch is None:
        metrics_run_epoch = ['accuracy']
    metrics = {
        'train': [],
        'test': [],
    }
    dataloader_train, dataloader_val, dataloader_test = dataloader

    # variables for early stopping
    best_model_state_dict = model.state_dict()
    metric_for_best_model = 'loss'
    metric_should_be_large = False
    early_stopping_coefficient = 1 if metric_should_be_large else -1
    best_value = -float('Inf') * early_stopping_coefficient
    steps_without_improvement = 0

    if dataloader_val is not None and use_validation_set:
        metrics['val'] = []

    criterion.to(device)
    parameter_container = criterion if hasattr(criterion, 'model') else model
    params = [p for p in parameter_container.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    lr_schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, verbose=True, patience=5),
    ]

    for epoch in range(num_epochs):
        string_length = str(len(str(num_epochs - 1)))
        prefix = ('CLEVER mind: Training the model {epoch:' + string_length + 'd}: ').format(epoch=epoch + 1)
        train_metrics = run_epoch(
            model, dataloader_train, criterion, optimizer=optimizer,
            train=True, metrics=metrics_run_epoch, metrics_dim=column_num, prefix=prefix+'Train',
            is_classification=is_classification)
        metrics['train'].append(train_metrics)

        verbose_string = prefix
        verbose_string += f'Train: {get_verbose_string(train_metrics)} | '

        if dataloader_val is not None and use_validation_set:
            val_metrics = run_epoch(
                model, dataloader_val, criterion,
                train=False, metrics=metrics_run_epoch, metrics_dim=column_num, prefix=prefix+'Val',
                is_classification=is_classification)
            metrics['val'].append(val_metrics)
            current_value = val_metrics[metric_for_best_model]
            lr_scheduler_metric = val_metrics['loss']

            verbose_string += f'Val: {get_verbose_string(val_metrics)} | '
        else:
            current_value = train_metrics[metric_for_best_model]
            lr_scheduler_metric = train_metrics['loss']
        for lr_scheduler in lr_schedulers:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(lr_scheduler_metric)
            else:
                lr_scheduler.step()

        if dataloader_test is not None:
            test_metrics = run_epoch(
                model, dataloader_test, criterion,
                train=False, metrics=metrics_run_epoch, prefix=prefix + 'Test',
                is_classification=is_classification)
            metrics['test'].append(test_metrics)
            verbose_string += f'Test: {get_verbose_string(test_metrics)}'

        tqdm.write(verbose_string)

        if early_stopping_coefficient * current_value > early_stopping_coefficient * best_value:
            best_value = current_value
            best_model_state_dict = model.state_dict()
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break

    model.load_state_dict(best_model_state_dict)
    return metrics, model

def pnn_fit_the_unknowns(
        model: nn.Module,
        dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader],
        device: str,
        criterion: Callable[[Tensor, Tensor], Tensor]=nn.CrossEntropyLoss(),
        weight_decay: float = 1e-5,
        is_classification: Union[bool, List[bool]] = True,
        learning_rate: float = 2e-3,
        use_validation_set: bool = True,
        num_epochs: int = 100,
        patience: int = 10,
        output_size: int = 2,
        metrics_run_epoch: Union[str, List[str]] = None) \
        -> Dict[str, List[Dict[str, float]]]:
    """Fitting the network for the unknown objects by adding a column``.

    Args:
        model: A PyTorch model
        dataloader:  A tuple of train-, val-, and test-dataloaders
        criterion: A function mapping the logits and targets to the loss
        weight_decay: The weight decay used in the optimizer
        device: Either cpu or cuda.
        column_num: n-th column of interest for updating the model.
        is_classification: A list of booleans indicating whether the metric
            is a classification metric or a regression metric (or a single
            boolean if all metrics are of the same type)
        learning_rate: The learning rate used in the Adam optimizer
        use_validation_set: Whether the training or validation set should be
            used for early stopping
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        output_size: The size of the output (we treat binary case; hence 2).
        metrics_run_epoch: Either 'all' or a list of metrics strings
            (Possible values: 'accuracy', 'brier score',
            'negative log likelihood', 'entropy', 'calibration',
            'entropy histogram')

    Returns:
        A dict containing the training, validation and test metrics
    """
    # enable training of only a column  
    model.add_new_column(is_classification=is_classification, output_size=output_size)

    if metrics_run_epoch is None:
        metrics_run_epoch = ['accuracy']
    dataloader_train, dataloader_val, dataloader_test = dataloader

    # variables for early stopping
    best_model_state_dict = model.state_dict()
    metric_for_best_model = 'loss'
    metric_should_be_large = False
    early_stopping_coefficient = 1 if metric_should_be_large else -1
    best_value = -float('Inf') * early_stopping_coefficient
    steps_without_improvement = 0

    metrics = {
        'train': [],
        'test': [],
    }
    if dataloader_val is not None and use_validation_set:
        metrics['val'] = []

    criterion.to(device)
    parameter_container = criterion if hasattr(criterion, 'model') else model
    params = [p for p in parameter_container.parameters() if p.requires_grad]
    optimizer = torch.optim.Adam(params, lr=learning_rate, weight_decay=weight_decay)
    lr_schedulers = [
        torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', factor=.5, verbose=True, patience=5),
    ]

    for epoch in range(num_epochs):
        string_length = str(len(str(num_epochs - 1)))
        prefix = ('CLEVER mind: Training the model {epoch:' + string_length + 'd}: ').format(epoch=epoch + 1)
        train_metrics = run_epoch(
            model, dataloader_train, criterion, optimizer=optimizer,
            train=True, metrics=metrics_run_epoch, prefix=prefix + 'Train',
            is_classification=is_classification)
        metrics['train'].append(train_metrics)

        verbose_string = prefix
        verbose_string += f'Train: {get_verbose_string(train_metrics)} | '

        if dataloader_val is not None and use_validation_set:
            val_metrics = run_epoch(
                model, dataloader_val, criterion,
                train=False, metrics=metrics_run_epoch, prefix=prefix + 'Val',
                is_classification=is_classification)
            metrics['val'].append(val_metrics)
            current_value = val_metrics[metric_for_best_model]
            lr_scheduler_metric = val_metrics['loss']

            verbose_string += f'Val: {get_verbose_string(val_metrics)} | '
        else:
            current_value = train_metrics[metric_for_best_model]
            lr_scheduler_metric = train_metrics['loss']
        for lr_scheduler in lr_schedulers:
            if isinstance(lr_scheduler, torch.optim.lr_scheduler.ReduceLROnPlateau):
                lr_scheduler.step(lr_scheduler_metric)
            else:
                lr_scheduler.step()

        if dataloader_test is not None:
            with torch.no_grad():
                test_metrics = run_epoch(
                    model, dataloader_test, criterion,
                    train=False, metrics=metrics_run_epoch, prefix=prefix + 'Test',
                    is_classification=is_classification)
            metrics['test'].append(test_metrics)
            verbose_string += f'Test: {get_verbose_string(test_metrics)}'

        tqdm.write(verbose_string)

        if early_stopping_coefficient * current_value > early_stopping_coefficient * best_value:
            best_value = current_value
            best_model_state_dict = model.state_dict()
            steps_without_improvement = 0
        else:
            steps_without_improvement += 1
            if steps_without_improvement >= patience:
                break

    model.load_state_dict(best_model_state_dict)
    return metrics, model

def bpnn_fit_the_knowns(
        model: TheCleverNetwork,
        dataloader: Tuple[DataLoader, DataLoader, DataLoader],
        train_dataset: Dataset,
        column_num: int,
        loss_function: nn.Module = nn.CrossEntropyLoss(),
        device: str = 'cuda',
        weight_decay: float = 1e-5,
        learning_rate: float = 2e-3,
        num_epochs: int = 1,
        patience: int = 10,
        curvature_num_samples: int = 1,
        is_force_convergence: bool = True,
        **kwargs) -> Dict[str, List[Dict[str, float]]]:
    """The dataset step of BPNN for fit_pnn.

    A new column is added and fitted. Moreover, the curvature is computed and
    the posterior is set.

    Args:
        model: A TheCleverNetwork object
        dataloader: A tuple of train-, val-, and test-dataloaders
        loss_function: The loss function to use
        weight_decay: The weight decay used in the BPNN
        learning_rate: The learning rate used to train the column
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        train_dataset: The training dataset (usually dataloader.dataset)
        criterion: The criterion that is used to optimize the parameters
        curvature_num_samples: The number of samples used to compute the curvature

    Returns:
        The metrics computed during the training
    """
    # load deterministic weights (init after loading)
    model.load_determinsitic_weights()

    # add new column
    if len(model.networks) == len(model.posterior):
        eval_num_samples = model.eval_num_samples
        model.eval_num_samples = 1

        if is_force_convergence:
            while True:
                train_metrics, model = pnn_fit_the_knowns(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    column_num=column_num,
                    criterion=loss_function,
                    weight_decay=weight_decay,
                    is_classification=[True for i in range(column_num+1)],
                    learning_rate=learning_rate,
                    use_validation_set=False,
                    num_epochs=num_epochs,
                    patience=patience
                    ) 
                if train_metrics['train'][0]['accuracy'] > 95.0:
                    break
        else:
            train_metrics, model = pnn_fit_the_knowns(
                model=model,
                dataloader=dataloader,
                device=device,
                column_num=column_num,
                criterion=loss_function,
                weight_decay=weight_decay,
                is_classification=[True for i in range(column_num+1)],
                learning_rate=learning_rate,
                use_validation_set=False,
                num_epochs=num_epochs,
                patience=patience
                ) 
        model.eval_num_samples = eval_num_samples
    else:
        raise AttributeError
    
    # save deterministic weights
    model.save_determinsitic_weights()

    # add new posterior
    model.current_column = column_num

    criterion = McAllesterCriterion(model=model, confidence=0.8)
    criterion.len_data = torch.as_tensor(len(train_dataset))
    criterion.loss_function = loss_function
    is_classification = type(loss_function) is nn.CrossEntropyLoss
    curvature_scaling = McAllesterScaling(
        confidence=0.8, 
        fixed=(None, None, None), 
        shared=(None, None, None))
    curvature_scaling.reset(
        dataloader=dataloader[1],
        criterion=criterion,
        is_classification=is_classification)
    
    # compute curvature and update posterior object
    model.update_existing_posterior(curvature_scaling, dataloader[0], 
                                    column_num=column_num, 
                                    num_samples=curvature_num_samples)

    # final tuning of the tempering
    model.tempering_a_posterior(column_num=column_num, dataloader=dataloader[1])
    return train_metrics, model

def bpnn_fit_the_unknowns(
        model: TheCleverNetwork,
        dataloader: Tuple[DataLoader, DataLoader, DataLoader],
        train_dataset: Dataset,
        loss_function: nn.Module = nn.CrossEntropyLoss(),
        device: str = 'cuda',
        output_size: int = 2,
        weight_decay: float = 1e-5,
        learning_rate: float = 2e-3,
        num_epochs: int = 1,
        patience: int = 10,
        curvature_num_samples: int = 1,
        is_force_convergence:bool = True,
        **kwargs) -> Dict[str, List[Dict[str, float]]]:
    """The dataset step of BPNN for fit_pnn.

    A new column is added and fitted. Moreover, the curvature is computed and
    the posterior is set.

    Args:
        model: A TheCleverNetwork object
        dataloader: A tuple of train-, val-, and test-dataloaders
        loss_function: The loss function to use
        output_size: The dimension of the network output
        weight_decay: The weight decay used in the BPNN
        learning_rate: The learning rate used to train the column
        num_epochs: The number of epochs
        patience: The number of epochs with no improvement after which training
            will be stopped
        train_dataset: The training dataset (usually dataloader.dataset)
        criterion: The criterion that is used to optimize the parameters
        curvature_num_samples: The number of samples used to compute the curvature

    Returns:
        The metrics computed during the training
    """
    # load deterministic weights (init after loading)
    model.load_determinsitic_weights()

    # add new column
    if len(model.networks) == len(model.posterior):
        eval_num_samples = model.eval_num_samples
        model.eval_num_samples = 1
        model.add_new_column(is_classification=True, output_size=output_size)
        if is_force_convergence:
            while True:
                train_metrics, model = pnn_fit_the_knowns(
                    model=model,
                    dataloader=dataloader,
                    device=device,
                    column_num=column_num,
                    criterion=loss_function,
                    weight_decay=weight_decay,
                    is_classification=[True for i in range(column_num+1)],
                    learning_rate=learning_rate,
                    use_validation_set=False,
                    num_epochs=num_epochs,
                    patience=patience
                    ) 
                if train_metrics['train'][0]['accuracy'] > 95.0:
                    break
        else:       
            train_metrics, model = pnn_fit_the_knowns(
                model=model,
                dataloader=dataloader,
                device=device,
                column_num=-1,
                criterion=loss_function,
                weight_decay=weight_decay,
                is_classification=[True for i in range(len(model.networks))],
                learning_rate=learning_rate,
                use_validation_set=False,
                num_epochs=num_epochs,
                patience=patience
                ) 

        model.eval_num_samples = eval_num_samples
    else:
        raise AttributeError
    
    # save deterministic weights
    model.save_determinsitic_weights()

    # add new posterior
    model.current_column = -1
    criterion = McAllesterCriterion(model=model, confidence=0.8)
    criterion.len_data = torch.as_tensor(len(train_dataset))
    criterion.loss_function = loss_function
    is_classification = type(loss_function) is nn.CrossEntropyLoss

    curvature_scaling = McAllesterScaling(
        confidence=0.8, 
        fixed=(None, None, None), 
        shared=(None, None, None))
    curvature_scaling.reset(
        dataloader=dataloader[1],
        criterion=criterion,
        is_classification=is_classification)
    
    model.add_new_posterior(curvature_scaling, dataloader[0], num_samples=curvature_num_samples)

    model.tempering_a_posterior(column_num=-1, dataloader=dataloader[1])
    return train_metrics, model