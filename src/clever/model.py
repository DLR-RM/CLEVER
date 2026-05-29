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
Model definition for clever system.
"""

import copy
import torch
import tqdm
import sys
import argparse
import numpy as np
import torchvision.transforms as transforms 

from torch import nn, Tensor
from torch.utils.data import DataLoader
from torch.nn import functional as F
from typing import List, Dict, Union, Optional
from mobile_sam import sam_model_registry, SamAutomaticMaskGenerator
from math import log

from bpnn.criterions import Criterion, PACBayesCriterion
from bpnn.curvature_scalings import CurvatureScaling
from bpnn.pnn import ProbabilisticProgressiveNeuralNetwork, ProgressiveNeuralNetwork
from bpnn.utils import device, fit, compute_curvature, mc_allester_bound, MetricSet
from curvature.curvatures import *
from clever.priors import compute_curvature_per_column

# TODO: fill in the docstrings.

class ClassifierNet(nn.Module):
    def __init__(self, input_size: int, output_size: int):
        super().__init__()
        self.input = nn.Linear(in_features=input_size, out_features=288)
        self.hidden_1 = nn.Linear(in_features=288, out_features=64)
        self.hidden_2 = nn.Linear(in_features=64, out_features=8)
        self.fc = nn.Linear(in_features=8, out_features=output_size)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        x = F.relu(self.hidden_1(x))
        x = F.relu(self.hidden_2(x))
        return self.fc(x)

def clever_pnn(args: argparse.ArgumentParser, 
               device: str, 
               last_layer_name: Optional[str] = 'fc', 
               lateral_connections: Optional[List[str]] = None):
    """_summary_

    Args:
        args (argparse.ArgumentParser): _description_
        device (str): _description_
        last_layer_name (Optional[str], optional): _description_. Defaults to 'fc'.
        lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

    Returns:
        _type_: _description_
    """
    # Load DINO backbone.
    try:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        dinov2.to(device).eval()
    except:
        dinov2 = torch.hub.load('/home_local/lee_jn/dinov2', 'dinov2_vits14_reg', source='local', pretrained=False)
        dinov2.load_state_dict(torch.load(args.model_path + '/dinov2/checkpoints/dinov2_vits14_reg4_pretrain.pth')) # TODO: no absolute path
        dinov2.to(device).eval()
        print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')
    
    # Load classifier
    mlp = ClassifierNet(input_size=list(dinov2.children())[-2].normalized_shape[0], output_size=2)
    classifier = copy.deepcopy(mlp).to(device)

    # Initialize PNN.
    model = ProgressiveNeuralNetwork(base_network=classifier,
                                     backbone=dinov2,
                                     last_layer_name=last_layer_name,
                                     lateral_connections=copy.deepcopy(lateral_connections))
    if args.model_pt_file is not None:
        model.load_full_state_dict(torch.load(args.model_pt_file))
    return model.to(device)

def pnn_predictions(pnn: nn.Module, image: torch.Tensor):
    """_summary_

    Args:
        pnn (nn.Module): _description_
        image (torch.Tensor): _description_

    Returns:
        _type_: _description_
    """
    output = pnn(image)
    return output

def mobile_sam(args: argparse.ArgumentParser, device: str, model_type: str = "vit_t"):
    """_summary_ # TODO: add some tuning functionalities

    Args:
        args (argparse.ArgumentParser): _description_
        device (str): _description_
        model_type (str, optional): _description_. Defaults to "vit_t".

    Returns:
        _type_: _description_
    """
    # load the model and make predictions
    sam_checkpoint = args.model_path + "./mobilesam/weight/mobile_sam.pt"
    sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
    sam.to(device=device)
    sam.eval()

    # obtain the mask generator
    mask_generator = SamAutomaticMaskGenerator(sam)
    return mask_generator

def sam_predictions(mask_generator: SamAutomaticMaskGenerator, image: np.array):
    """_summary_ # TODO: parse the masks

    Args:
        mask_generator (SamAutomaticMaskGenerator): _description_
        image (np.array): _description_

    Returns:
        _type_: _description_
    """
    masks = mask_generator.generate(image)
    return masks


class TheCleverNetwork(ProbabilisticProgressiveNeuralNetwork):
    """ The Clever Network for stream-based active learning.

    This network architecture builds on top of Progressive Neural Networks and
    computes a posterior for the layers. For this, it utilizes a prior
    computed on a different related dataset and computes the posterior with
    Laplace approximation.

    The prior for the main columns is the posterior of the prior task. There
    will not be any lateral connections for transfer between each columns.
    This is due to the construction of the problem - learning to classify
    an apple may not directly lead to positive transfers.

    The steps of training a Bayesian Progressive Neural Network are
    
    1. Compute the posterior on a related dataset (init_prior).
    The prior is learned from synthetic dataset.
    
    2. For each incoming task:
        1. Add a new column (add_new_column)
        2. Fit weights of last column and incoming lateral connections to the
           dataset
        3. Compute the curvature on the dataset (compute_new_curvature)
        4. Combine the prior curvature and new curvature of the likelihood with
           a curvature scaling to obtain the new posterior (add_new_posterior)
    
    3. For updating existing task:
        1. Fit training data for new weights
        2. Compute the curvature on the new dataset
        3. Combine the prior curvature and new curvature of the likelihood with
           a curvature scaling to obtain the new posterior (add_new_posterior)
    """   
    def __init__(
            self,
            model: nn.Module,
            priors: List[Curvature] = None,
            backbone: nn.Module = None,
            last_layer_name: Optional[str] = 'fc',
            lateral_connections: Optional[List[str]] = None,
            train_resample_slice: slice = slice(-1),
            train_num_samples: int = 1,
            eval_resample_slice: slice = slice(None),
            eval_num_samples: int = 100,
            weight_decay: float = 1e-5,
            weight_decay_layer_names: Optional[List[str]] = None,
            curvature_device: Optional[torch.device] = torch.device('cpu'),
            curvature_scaling_device: Optional[torch.device] = torch.device('cpu'),
            pac_bayes_iterations: int = 100
            ):
        """TheCleverNetwork initializer.

        Args:
            prior: A Curvature (that is inverted) that is used as prior
            backbone: A PyTorch model that is used as a backbone for the
                Progressive Neural Network. If not provided, no backbone is used.
            last_layer_name: The name of the last layer
            lateral_connections: The names of the layers that should have
                lateral connections
            train_resample_slice: The columns from which the weight should be
                sampled during training
            train_num_samples: The number of samples used for each forward pass
                during training
            eval_resample_slice: The columns from which the weight should be
                sampled during evaluation
            eval_num_samples: The number of samples used for each forward pass
                during evaluation
            weight_decay: The weight decay used for the prior
            weight_decay_layer_names: Layers that use an isotropic Gaussian
                prior with default_weight_decay (should usually include
                last_layer_name when the columns have different number of
                outputs)
            curvature_device: The device for the curvature
            curvature_scaling_device: The device used for the curvature scaling
        """
        super().__init__(
            model.base_network, backbone, last_layer_name, lateral_connections,
            train_resample_slice, train_num_samples, eval_resample_slice, eval_num_samples)
        
        self.posterior = None
        if priors is not None:
            assert len(model.networks) == len(priors)

            # updates the names of the state dict
            for prior in priors:
                prior.model_state = prior.model.state_dict()
            # approximated posterior curvatures (are needed for sampling and KL-divergence computation)
            self.posterior = priors
            self.networks = model.networks

        self.weight_decay = weight_decay
        self.scales = {}
        self.temperature_scaling = {}

        self.weight_decay_layer_names = weight_decay_layer_names

        self.curvature_device = curvature_device
        self.curvature_scaling_device = curvature_scaling_device

        self.fixed_model_size_trace = 0.
        self.current_column = -1
        self.pac_bayes_iterations = pac_bayes_iterations

        self.is_classification = [True for i in range(len(model.networks))]

        self.deterministic_weights_dict = {}

    @property
    def previous_tasks(self):
        """The number of previously finished tasks.

        Different from Progressive Neural Networks as the task is only finished
        when the posterior is available.

        Returns:
            The number of previously finished tasks.
        """
        return len(self.posterior) + 1

    def save_determinsitic_weights(self):
        """ Saves determinsitic parameters into a dictionary
        """
        self.deterministic_weights_dict = copy.deepcopy(super().full_state_dict())

    def load_determinsitic_weights(self):
        """ Loads determinsitic parameters into a dictionary
        """
        if len(self.deterministic_weights_dict) != 0:
            self.last_layer_name = self.deterministic_weights_dict['last_layer_name']
            self.base_network.load_state_dict(self.deterministic_weights_dict['base_network'])
            self.backbone.load_state_dict(self.deterministic_weights_dict['backbone'])
            self.networks.load_state_dict(self.deterministic_weights_dict['networks'])

    def full_state_dict(self) -> Dict[str, Any]:
        """Returns a dictionary containing the whole state.

        Returns:
            A dictionary containing the whole state
        """
        full_state_dict = super().full_state_dict()

        full_state_dict.update(
            {
                'posterior': [
                    posterior.state_dict() for posterior in self.posterior
                ] if self.posterior is not None else None,
                'weight_decay': self.weight_decay,
                'scales':
                    [[self.posterior[i].alpha, self.posterior[i].beta]
                    for i in range(len(self.networks))],
                'temperature_scaling':
                    [self.posterior[i].temperature_scaling
                    for i in range(len(self.networks))],
                'weight_decay_layer_names': self.weight_decay_layer_names,
                'fixed_model_size_trace': self.fixed_model_size_trace
            }
        )
        return full_state_dict

    def load_full_state_dict(self, full_state_dict):
        """Copies the whole state from full_state_dict.

        Args:
            full_state_dict: A dict containing the full state
        """
        super().load_full_state_dict(full_state_dict)
        self.weight_decay = full_state_dict['weight_decay']
        self.weight_decay_layer_names = full_state_dict['weight_decay_layer_names']
        self.fixed_model_size_trace = full_state_dict['fixed_model_size_trace']

        if full_state_dict['networks']:
            if full_state_dict['posterior'] is not None:
                self.posterior = []
                for curvature_states, network in zip(full_state_dict['posterior'], self.networks):
                    curvature_column = eval(curvature_states['class'])(network)
                    curvature_column.remove_hooks_and_records()
                    curvature_column.load_state_dict(curvature_states)
                    self.posterior.append(curvature_column)

            # loading hyperparameters
            counter = 0
            for scale, temperature_scaling in zip(full_state_dict['scales'], full_state_dict['temperature_scaling']):
                self.posterior[counter].alpha = scale[0]
                self.posterior[counter].beta = scale[1]
                self.posterior[counter].temperature_scaling = temperature_scaling
                counter = counter + 1
            self.scale = scale 
            self.temperature_scaling = temperature_scaling
            self.networks[-1].requires_grad_(True)
    
    def add_new_column(
            self,
            is_classification: bool = True,
            output_size: Optional[int] = 2,
            differ_from_previous: bool = False,
            resample_base_network: bool = False):
        """Adds a new column.

        Args:
            is_classification: Whether the new column is a classification or a
                regression task.
            output_size: The dimension of the output of the last layer of the
                new column (is usually the number of classes in the
                corresponding dataset)
            differ_from_previous: Whether the new weights should be altered
                slightly
            resample_base_network: Whether the weights should be resampled
                completely (new random initialization)
        """
        super().add_new_column(is_classification, output_size, differ_from_previous, resample_base_network)

    def update_scaling(
            self,
            alpha: Optional[Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            beta: Optional[Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            temperature_scaling: Optional[
                Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None,
            update_slice: Optional[slice] = slice(-1, None)):
        """Updates the scales and temperature scaling.

        Args:
            alpha: The new value for the scale of the prior.
            beta: The new value for the scale of the likelihood.
            temperature_scaling: The new value for the temperature scaling.
            update_slice: The slice of the networks to be updated only used if the
                scalar is a number.
        """
        if alpha is not None and beta is not None:
            for i, scaling in enumerate([alpha, beta]):
                if scaling is not None:
                    if isinstance(scaling, dict):
                        for module, scale in scaling.items():
                            if module not in self.scales.keys():
                                self.scales[module] = [None, None]
                            self.scales[module][i] = scale
                    else:
                        raise TypeError

        if temperature_scaling is not None:
            if isinstance(temperature_scaling, dict):
                for module, scale in temperature_scaling.items():
                    self.temperature_scaling[module] = scale
            else:
                raise TypeError

    def update_for_new_posterior(
            self,
            alpha: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            beta: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            curvature: Curvature,
            isotropic: Curvature):
        """Updates the posterior for a new column.

        This method sets the posterior curvature to
            alpha * prior + beta * curvature.

        Prior here is an isotropic prior since we added it for a new task
        with no synthetic data.

        Args:
            alpha: The prior scale
            beta: The likelihood scale
            curvature: The new curvatures to be used for the update
        """
        # updating and appending the posterior
        self.posterior[-1] = curvature.add_and_scale(
            isotropic, 
            [beta, alpha], 
            self.weight_decay_layer_names, self.weight_decay)
        
        assert len(self.networks) == len(self.posterior)

        # inversion
        self.posterior[-1].invert()
       
    def add_new_posterior(
            self,
            curvature_scaling: CurvatureScaling,
            dataloader: DataLoader,
            num_samples: int = 1,
            return_curvature: bool = False):
        """Adds a new posterior.

        Args:
            curvature_scaling: The curvature scaling used to determine
                alpha and beta
            dataloader: The dataloader that is used to compute the curvature
            num_samples: The number of samples used to compute the curvature
            return_curvature: Whether the curvature should be returned
        """
        assert len(self.networks) == len(self.posterior) + 1, \
            'compute_new_curvature and add_new_column should be called before add_new_posterior'
        
        # new curvature after MAP training for the new task
        new_curvature = KFOC(self.networks[-1], layer_types='Linear', device=self.curvature_device)

        self.posterior.append([])

        # compute the curvature
        negative_data_log_likelihood = compute_curvature_per_column(
                model=self,
                dataloader=dataloader,
                column_num=-1,
                curvs=[new_curvature],
                return_data_log_likelihood=True,
                num_samples=num_samples,
                invert=True,
                make_positive_definite=False,
                categorical=True,
            )
        
        # remove the hooks and record after computations        
        new_curvature.remove_hooks_and_records()

        # defining an isotropoic prior
        isotropic = new_curvature.eye_like(weight_decay=self.weight_decay)
        isotropic.remove_hooks_and_records()
        
        # define a tracable function
        def update(alpha, beta, temperature_scaling):
            global new_posterior
            new_posterior = new_curvature.add_and_scale(isotropic, [beta, alpha])
            new_posterior.invert()

        def trace(temperature_scaling):
            global new_posterior
            return new_curvature.trace_of_mm(
                new_posterior,
                temperature_scaling=temperature_scaling
            )

        def kl_divergence(temperature_scaling):
            global new_posterior
            return new_posterior.kl_divergence(
                isotropic,
                temperature_scaling=temperature_scaling
            )
        
        # assigning to the curvature scaling device, e.g., cpu
        new_curvature.to(self.curvature_scaling_device)
        isotropic.to(self.curvature_scaling_device)

        # assign the values for the quadratic terms
        self.prior_quadratics = isotropic

        # find optimal parameters
        alpha, beta, temperature_scaling = curvature_scaling.find_optimal_scaling(
            update=update,
            model=self,
            trace=trace,
            kl_divergence=kl_divergence,
            modules=new_curvature.state.keys(),
            len_data=len(dataloader.dataset),
            negative_data_log_likelihood=negative_data_log_likelihood,
            scaled_fixed_model_size=self.fixed_model_size_trace,
            device=self.curvature_scaling_device,
            max_iter=self.pac_bayes_iterations
        )

        # assigning to the curvature device, e.g., cpu
        new_curvature.to(self.curvature_device)

        # updating the posteriors
        self.update_for_new_posterior(alpha, beta, new_curvature, isotropic)
        self.update_scaling(alpha=alpha, beta=beta, temperature_scaling=temperature_scaling)
        self.fixed_model_size_trace += self.trace(new_curvature, self.posterior[-1]).item()

        # add hyperparameters to curvature object of the posterior
        self.posterior[-1].alpha = alpha
        self.posterior[-1].beta = beta
        self.posterior[-1].temperature_scaling = temperature_scaling

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_curvature:
            return new_curvature, negative_data_log_likelihood

    def get_quadratic_term(self):
        """Computes the quadratic term from the normal prior and posterior.

        Let :math:`\\theta` be the parameters of the column and
        :math:`\\hat{\\theta}` be the parameters and :math:`\\Sigma` the
        curvature of the prior, then this method computes
         .. math:
            \frac{1}{\\tau}(\\theta - \\hat{\\theta})^T \\Sigma (\\theta - \\hat{\\theta})
        :math:`\\Sigma_l = weight_decay I` for all layers in weight_decay_layer_names.

        Returns:
            A tensor containing the quadratic term (# NOTE: are you using them when computing just a normal loss?)
        """ # TODO: why model_qaudratics in the module list?
        out = torch.as_tensor(0., device=device)
        out += self.prior_quadratics.get_quadratic_term(
            self.networks[self.current_column],
            weight_decay_layer_names=self.weight_decay_layer_names,
            default_weight_decay=self.weight_decay)
        return out

    def kl_divergence(self, posterior: Curvature, prior: Curvature, temperature_scaling=None):
        """Computes the Kullback-Leibler(KL)-divergence.

        The KL-divergence is between posterior and prior.

        Args:
            posterior: Curvatures for the likelihood term
            prior: Curvature from the prior term
            temperature_scaling: The temperature scaling

        Returns:
            A tensor containing the KL-divergence
        """
        if temperature_scaling is None:
            temperature_scaling = self.temperature_scaling

        out = torch.as_tensor(0., device=device)
        out += posterior.kl_divergence(prior,
                temperature_scaling=temperature_scaling,
                weight_decay_layer_names=self.weight_decay_layer_names,
                default_weight_decay=self.weight_decay)
        return out

    def trace(
            self,
            posterior: Curvature,
            prior: Curvature,
            temperature_scaling: Optional[
                Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]]] = None):
        """Computes the trace of the matrix product of the Fisher matrix in curvature_list
        and the last posterior.

        Args:
            posterior: Curvatures for the likelihood term
            prior: Curvature from the prior term
            temperature_scaling: The temperature scaling

        Returns:
            A tensor containing the trace
        """ 
        if temperature_scaling is None:
            temperature_scaling = self.temperature_scaling

        out = torch.as_tensor(0., device=device)
        out += posterior.trace_of_mm(prior, temperature_scaling=temperature_scaling)
        return out
    
    def update_for_existing_posterior(
            self,
            alpha: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            beta: Union[float, torch.Tensor, Dict[nn.Module, Union[float, torch.Tensor]]],
            column_num: int,
            curvature: Curvature,
            prior: Curvature):
        """Updates the posterior for a new column.

        This method sets the posterior curvature to
            alpha * prior + beta * curvature.

        Prior here is an isotropic prior since we added it for a new task
        with no synthetic data.

        Args:
            alpha: The prior scale
            beta: The likelihood scale
            column_num: column number
            curvature: The new curvatures to be used for the update
            prior: prior from the previous posterior
        """
        # updating and appending the posterior
        self.posterior[column_num] = curvature.add_and_scale(
            prior, [beta, alpha], self.weight_decay_layer_names, self.weight_decay
            )
        assert len(self.networks) == len(self.posterior)

        # inversion
        self.posterior[column_num].invert()

    def update_existing_posterior(
            self,
            curvature_scaling: CurvatureScaling,
            dataloader: DataLoader,
            column_num: int,
            num_samples: int = 1,
            return_curvature: bool = False):
        """Adds a new posterior.

        Args:
            curvature_scaling: The curvature scaling used to determine
                alpha and beta
            dataloader: The dataloader that is used to compute the curvature
            column_num: The column number to update
            num_samples: The number of samples used to compute the curvature
            return_curvature: Whether the curvature should be returned
        """
        assert len(self.networks) == len(self.posterior), \
            'number of columns and posterior dimensions has to match!'
        
        # new curvature after MAP training for the new task
        new_curvature = KFOC(self.networks[column_num], layer_types='Linear', device=self.curvature_device)

        # compute the curvature        
        negative_data_log_likelihood = compute_curvature_per_column(
                model=self,
                dataloader=dataloader,
                column_num=column_num,
                curvs=[new_curvature],
                return_data_log_likelihood=True,
                num_samples=num_samples,
                invert=True,
                make_positive_definite=False,
                categorical=True,
            )
        
        # remove the hooks and record after computations        
        new_curvature.remove_hooks_and_records()

        # defining an isotropoic prior / posterior as prior
        prior = self.posterior[column_num]
        prior.remove_hooks_and_records()
        
        # define a tracable function
        def update(alpha, beta, temperature_scaling):
            global new_posterior
            new_posterior = new_curvature.add_and_scale(prior, [beta, alpha])
            new_posterior.invert()

        def trace(temperature_scaling):
            global new_posterior
            return new_curvature.trace_of_mm(
                new_posterior,
                temperature_scaling=temperature_scaling
            )

        def kl_divergence(temperature_scaling):
            global new_posterior
            return new_posterior.kl_divergence(
                prior,
                temperature_scaling=temperature_scaling
            )
        
        # assigning to the curvature scaling device, e.g., cpu
        new_curvature.to(self.curvature_scaling_device)
        prior.to(self.curvature_scaling_device)

        # assign the values for the quadratic terms
        self.prior_quadratics = prior

        # find optimal parameters
        alpha, beta, temperature_scaling = curvature_scaling.find_optimal_scaling(
            update=update,
            model=self,
            trace=trace,
            kl_divergence=kl_divergence,
            modules=new_curvature.state.keys(),
            len_data=len(dataloader.dataset),
            negative_data_log_likelihood=negative_data_log_likelihood,
            scaled_fixed_model_size=0,
            device=self.curvature_scaling_device,
            max_iter=self.pac_bayes_iterations
        )

        # assigning to the curvature device, e.g., cpu
        new_curvature.to(self.curvature_device)

        # updating the posteriors
        self.update_for_existing_posterior(alpha, beta, column_num, new_curvature, prior)
        self.update_scaling(alpha=alpha, beta=beta, temperature_scaling=temperature_scaling)
        self.fixed_model_size_trace += self.trace(new_curvature, self.posterior[column_num]).item()

        # add hyperparameters to curvature object of the posterior
        self.posterior[column_num].alpha = alpha
        self.posterior[column_num].beta = beta
        self.posterior[column_num].temperature_scaling = temperature_scaling

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if return_curvature:
            return new_curvature, negative_data_log_likelihood

    def forward(self, x: Tensor):
        assert self.networks, 'no column is available, please call add_new_column before forward_once'
        x = self.backbone(x)
        out = [column[-1](x) for column in self.networks]
        self.forward_record.clear()
        return out
    
    def _sample_and_replace(self, resample_slice: slice):
        # Note that resample_slice is the slice of the posterior,
        # but there might be more columns than posteriors

        # implement sampling and replacing by the posterior 
        # NOTE: think, is this the fastest way?
        for _, column in zip(
                self.networks[resample_slice],
                self.posterior):
            column.sample_and_replace(temperature_scaling=column.temperature_scaling)

    def _stochastic_forward_once(
            self,
            feature: Tensor):
        self.sample_and_replace(resample_slice=None)
        return self._column_forward_once(feature)
    
    def _column_forward_once(self, features: Tensor):
        assert self.networks, 'no column is available, please call add_new_column before forward_once'
        out = [column[-1](features) for column in self.networks]
        self.forward_record.clear() # NOTE: why are we -1 here?
        return out

    def bayesian_analysis(self, x: Tensor, num_samples: Optional[int] = 30):
        assert self.networks, \
            'no column is available, please call add_new_column before forward'
        if num_samples is None:
            num_samples = self.train_num_samples if self.training else self.eval_num_samples
        assert num_samples >= 1, \
            f'num_samples should be larger or equal than 1, but it is {num_samples}'
        # feature computations
        features = self.backbone(x)

        # MC-sampling over smaller network
        logits = [self._stochastic_forward_once(features)
                  for _ in range(num_samples)]
        out = [torch.stack([l[j] for l in logits], dim=2)
               for j in range(len(self.networks))]
        return out
    
    def evaluate_per_column(self, model: nn.Module, column_num: int, dataloader: DataLoader):
        """_summary_

        Args:
            model (nn.Module): _description_
            column_num (int): _description_
            dataloader (DataLoader): _description_

        Returns:
            _type_: _description_
        """
        model.eval()
        criterion = nn.CrossEntropyLoss()

        if column_num == -1:
            is_classification = [True for i in range(len(model.networks))]
        else:
            is_classification = [True for i in range(column_num+1)]

        metrics_obj = MetricSet(
            len(dataloader.dataset), 'all', column_num,
            is_classification)

        mean_loss = torch.as_tensor(0., device=device)
        model.to(device)

        with torch.no_grad():
            for (features, targets) in dataloader:
                features, targets = features.to(device), targets.to(device)
                logits = model.bayesian_analysis(features) # model(features)
                logits = [logit.permute(0, 2, 1) for logit in logits]
                logit = logits[column_num].permute(0, 2, 1)

                probs = F.softmax(logit, dim=-1)
                loss = criterion(probs.mean(dim=1).log(), targets)

                mean_loss += loss.detach() / len(dataloader)
                metrics_obj.update([logit.squeeze().detach() for logit in logits], targets.detach())

            metrics_dict = metrics_obj.summarize()
            metrics_dict['loss'] = mean_loss.item()

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        return metrics_dict
    
    def tempering_a_posterior(self, column_num: int, dataloader: DataLoader, temperature_scalings: List = None):
        """_summary_

        Args:
            column_num (int): _description_
            dataloader (DataLoader): _description_
            temperature_scalings (List, optional): _description_. Defaults to None.

        Raises:
            NotImplementedError: _description_
        """ 
        # TODO: maybe bayesian optimization can be an option here.
        # currently a basic implementation of iterating some values 
        # and saving the most promising results as a final outcome
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        if temperature_scalings is None:
            temperature_scalings = \
            [1e-0, 1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6, 1e-7, 1e-8, 1e-9, 1e-10, 1e-11, 1e-12, 1e-13, 1e-14, 1e-15, 1e-25, 1e-50]
            # temperature_scalings = \
            #     [1e-2, 1e-3, 1e-5, 1e-6, 1e-7, 1e-9, 1e-10, 1e-50]
        
        current_tempering = copy.deepcopy(self.posterior[column_num].temperature_scaling)

        result = []
        accuracy = []
        ece = []
        for a_temperature_scaling in temperature_scalings:
            # multiply all the tempering values and assign to the current posterior of interest
            tempering = {k: np.multiply(v, a_temperature_scaling) for k, v in current_tempering.items()}
            self.posterior[column_num].temperature_scaling = tempering

            # evaluate bayesian analysis per column and collect
            metric = self.evaluate_per_column(model=self, column_num=column_num, dataloader=dataloader)
            measure = (100.0-metric['accuracy']) + metric['calibration'][0]*100

            # collect the results
            result.append(measure)
            accuracy.append(metric['accuracy'])
            ece.append(metric['calibration'][0])

        # pick a quantile based temperature
        index_max = np.argwhere(result == np.amin(result))
        a_temperature_scaling = temperature_scalings[index_max[0][0]]
        tempering = {k: np.multiply(v, a_temperature_scaling) for k, v in current_tempering.items()}
        # TODO: here, it could be that you need to reset the tempering mechanism.
        self.posterior[column_num].temperature_scaling = tempering

        print("CLEVER mind: Selecting tempering parameter ", a_temperature_scaling, " with accuracy(%) ", accuracy[index_max[0][0]], " and ECE(%) ", ece[index_max[0][0]]*100)

        if torch.cuda.is_available():
            torch.cuda.empty_cache()


class McAllesterCriterion(PACBayesCriterion):
    """Uses the McAllester bound."""

    def forward(self, logits, targets):
        nll = self.loss_function(logits, targets)
        trace_fixed = self.model.fixed_model_size_trace
        num_parameters_current = sum(p.numel() for p in self.model.networks[self.model.current_column].parameters())
        expected_empirical_risk = (nll + (trace_fixed + num_parameters_current) / (2 * self.len_data)) / log(2)
        quadratic_term = .5 * self.model.get_quadratic_term()
        return mc_allester_bound(expected_empirical_risk, quadratic_term, self.len_data, self.confidence)
