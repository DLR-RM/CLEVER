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
import sys

from tqdm import tqdm
from torch import nn
from torch.utils.data import DataLoader, Dataset

from curvature.curvatures import *
from bpnn.curvature_scalings import CurvatureScaling, McAllesterScaling
from bpnn.utils import device, fit, compute_curvature, HalfMSELoss
from curvature.utils import seed_all_rng

seed = seed_all_rng()
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
base_path = os.path.normpath(os.path.join(os.path.dirname(os.path.realpath(__file__)), os.pardir, os.pardir))
torch_data_path = os.path.join(base_path, 'data', 'torch')

def init_clever_prior(model: nn.Module, dataloaders: List, weight_decay: float):
    """_summary_

    Args:
        model (nn.Module): _description_
        dataloaders (List): _description_
        weight_decay (float): _description_

    Returns:
        priors (List[Curvature]): _description_
    """
    # iterate per column with synthetic dataset
    priors = list()
    scaling_method = McAllesterScaling(confidence=0.8, fixed=(None, None, None), shared=(None, None, None))
    for n_column in range(len(dataloaders)-1):
        prior = init_prior_per_column(base_network=model, 
                                      curvature_type=KFOC,
                                      dataloader=dataloaders[n_column+1][1][0],
                                      weight_decay=weight_decay,
                                      curvature_scaling=scaling_method,
                                      column_num=n_column)
        priors.append(prior)
    return priors

def init_prior_per_column(
        base_network: nn.Module,
        curvature_type: type,
        dataloader: DataLoader,
        weight_decay: float,
        curvature_scaling: CurvatureScaling,
        column_num: int,
        isotropic_prior: bool = False,
        curvature_device: Optional[torch.device] = None,
        curvature_scaling_device: Optional[torch.device] = None,
        curvature_path: str = None,
        negative_data_log_likelihood: float = None,
        len_data: int = None
        ) -> Curvature:
    """Initializes the prior.

    It computes the curvature around the base_network using the dataloader and
    returns a curvature containing the prior distribution. If curvature_path is
    provided, the curvature is loaded from the path (also requires the
    negative_data_log_likelihood and len_data). If isotropic_prior is True, the
    prior is an isotropic Gaussian with weight_decay as variance.

    If the curvature is computed and not isotropic, we scale it using the
    curvature_scaling.

    Args:
        base_network: The base network
        curvature_type: The class of curvature to be used
        dataloader: The dataloader to compute the curvature
        weight_decay: The weight decay used for the prior
        column_num: Nth column of interest (starting from zero)
        curvature_scaling: The curvature scaling used to scale the prior
        isotropic_prior: Whether the prior should be isotropic
        curvature_device: The device used for the curvature
        curvature_scaling_device: The device used for the curvature scaling
        curvature_path: The path to the curvature
        negative_data_log_likelihood: The negative data log likelihood of the
            curvature
        len_data: The number of data points used to compute the curvature

    Returns:
        The prior
    """
    if curvature_device is None:
        curvature_device = device
    
    prior_curvature = curvature_type(base_network.networks[column_num], 
                                     layer_types='Linear', 
                                     device=curvature_device)

    if curvature_path and negative_data_log_likelihood and len_data:
        curvature_state_dict = torch.load(curvature_path, map_location=curvature_device)
        prior_curvature.load_state_dict(curvature_state_dict)
    else:
        if isotropic_prior:
            negative_data_log_likelihood = compute_curvature_per_column(
                model=base_network,
                dataloader=[next(iter(dataloader))],
                column_num=column_num,
                curvs=[prior_curvature],
                return_data_log_likelihood=True,
                num_samples=1,
                invert=True,
                make_positive_definite=False,
                device=curvature_device,
                categorical=True,
            )
        else:
            negative_data_log_likelihood = compute_curvature_per_column(
                model=base_network,
                dataloader=dataloader,
                column_num=column_num,
                curvs=[prior_curvature],
                return_data_log_likelihood=True,
                num_samples=1,
                invert=True,
                make_positive_definite=False,
                device=curvature_device,
                categorical=True,
            )
            len_data = len(dataloader.dataset)

    prior_curvature.remove_hooks_and_records()

    isotropic = prior_curvature.eye_like(weight_decay=weight_decay)
    isotropic.remove_hooks_and_records()

    if isotropic_prior:
        return isotropic

    for curvature in [isotropic, prior_curvature]:
        curvature.to(curvature_scaling_device)

    def update(alpha, beta, temperature_scaling):
        global prior
        prior = prior_curvature.add_and_scale(isotropic, [beta, alpha])
        prior.invert()

    def trace(temperature_scaling):
        global prior
        return prior_curvature.trace_of_mm(
            prior,
            temperature_scaling=temperature_scaling
        )

    def kl_divergence(temperature_scaling):
        global prior
        return prior.kl_divergence(
            isotropic,
            temperature_scaling=temperature_scaling
        )
    
    alpha, beta, temperature_scaling = curvature_scaling.find_optimal_scaling(
        update=update,
        model=base_network,
        trace=trace,
        kl_divergence=kl_divergence,
        modules=prior_curvature.state.keys(),
        len_data=len_data,
        negative_data_log_likelihood=negative_data_log_likelihood,
        scaled_fixed_model_size=0,
        device=curvature_scaling_device,
        max_iter=5000
    )
    update(alpha, beta, temperature_scaling)

    prior.alpha = alpha
    prior.beta = beta
    prior.temperature_scaling = temperature_scaling
    prior.scale_inverse(temperature_scaling)

    for curvature in [isotropic, prior_curvature, prior]:
        curvature.to(curvature_device)

    base_network.networks[column_num].load_state_dict(prior.model_state)
    return prior

def compute_curvature_per_column(
        model: nn.Module,
        curvs: List[Curvature],
        dataloader: DataLoader,
        column_num: int,
        return_data_log_likelihood: bool = True,
        num_samples: int = 1,
        invert: bool = False,
        make_positive_definite: bool = False,
        device: torch.device = device,
        seed: int = seed,
        categorical: bool = True,
        save_path: str = None,
        save_every_n_steps: int = 100):
    """Computes the curvatures.

    Changes the curvatures and can return the negative log-likelihood.

    Args:
        model: A PyTorch model
        curvs: A list of curvature objects corresponding to ``model``
        dataloader: A PyTorch dataloader
        column_num: Nth column of interest (starting from zero)
        return_data_log_likelihood: Whether the negative log-likelihood should
            be computed and returned
        num_samples: The number of samples used to compute the curvature
        invert: Whether the curvature should be inverted afterwards
        make_positive_definite: Whether the curvature should be made positive
            definite
        device: A device
        seed: A seed
        categorical: Whether the targets are categorical or continuous
        save_path: Path where intermediate and final results of the curvature
            should be saved (or not saved at all if ``save_path = None``
        save_every_n_steps: The frequency of saving the curvature

    Returns:
        negative log-likelihood if ``return_data_log_likelihood`` else None
    """
    # assign models to get potential gradients
    # enable training of only a column  
    model.to(device)
    model.base_network.requires_grad_(False)
    model.base_network.eval()
    model.networks.requires_grad_(False)
    model.networks.eval()

    intra_column = [column for column in model.networks]
    intra_column[column_num].requires_grad_(True)
    intra_column[column_num].train()

    generator = torch.Generator('cpu').manual_seed(seed)
    criterion = nn.CrossEntropyLoss() if categorical else HalfMSELoss()
    log_likelihood_criterion = nn.CrossEntropyLoss(reduction='sum') if categorical \
        else HalfMSELoss(reduction='sum')
    log_likelihood = 0.

    for curv in curvs: 
        curv.model.requires_grad_(False)
        curv.model.eval()

        intra_column = [column for column in curv.model]
        intra_column[column_num].requires_grad_(True)
        intra_column[column_num].train()

    with tqdm(dataloader, file=sys.stdout) as t:
        t.set_description("CLEVER mind: Performing Bayesian inference! ")
        for i, (inputs, targets) in enumerate(t):
            if save_path is not None and i % save_every_n_steps == 0:
                torch.save([curv.state_dict() for curv in curvs], save_path)

            inputs, targets = inputs.to(device), targets.to(device)
            features = model.backbone(inputs)
            logits = [column[-1](features) for column in model.networks][column_num]
            model.forward_record.clear()

            if isinstance(logits, list):
                logits = logits[-1]

            if logits.ndim == 3:
                if categorical:
                    if logits.shape[1] > 1:
                        probs = F.softmax(logits, dim=-1)
                        logits = probs.mean(dim=1).log()
                    else:
                        logits = logits[:, 0, :]
                else:
                    logits = logits.mean(dim=1)

            if return_data_log_likelihood:
                log_likelihood += log_likelihood_criterion(logits, targets).detach().item()

            for _ in range(num_samples):
                generator_local = torch.Generator(device) \
                    .manual_seed(torch.randint(0, 0xffff_ffff, [], generator=generator).item())
                sampled_labels = torch.multinomial(F.softmax(logits, dim=1), 1, True, generator=generator_local)[:, 0] \
                    if categorical else torch.normal(logits, 1., generator=generator_local)

                loss = criterion(logits, sampled_labels)
                model.zero_grad()
                loss.backward(retain_graph=num_samples > 1)

                for curv in curvs:
                    curv.update(batch_size=features.size(0))

    for curv in curvs:
        curv.remove_hooks_and_records()
        curv.scale(num_batches=len(dataloader) * num_samples, make_pd=make_positive_definite)
        if invert:
            curv.invert()

    if save_path is not None:
        torch.save([curv.state_dict() for curv in curvs], save_path)

    if return_data_log_likelihood:
        return log_likelihood
    
