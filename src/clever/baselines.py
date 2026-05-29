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
Implementation of baselines for CLEVER paper.
"""

import torch
import torch.nn.functional as F
import argparse
import copy
import gc
import os
import cv2
import sys
import numpy as np

from typing import List, Dict, Optional, Any, Callable, Tuple
from scipy.stats import entropy
from torch.utils.data import DataLoader

from curvature.curvatures import *
from bpnn.curvature_scalings import CurvatureScaling, McAllesterScaling
from bpnn.pnn import ProgressiveNeuralNetwork, DropoutProgressiveNeuralNetwork
from clever.trainer import pnn_fit_the_knowns, pnn_fit_the_unknowns, bpnn_fit_the_knowns, bpnn_fit_the_unknowns
from clever.model import ClassifierNet, clever_pnn, TheCleverNetwork
from clever.data.clever import CleverDataset
from clever.clever import TheSystem
from clever.priors import init_prior_per_column


class Dropout(TheSystem):
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 device: str,
                 sam_gap: int = 1000,
                 num_seg: int = 4,
                 is_temporal: bool = False,
                 class_mapping: dict = {"apple": 0},
                 mc_samples: int = 10,
                 pairwise_coupling: str = "normalization"):
        super().__init__(
            args, device, sam_gap, num_seg,
            is_temporal, class_mapping, mc_samples, pairwise_coupling)
        self.class_mapping = {}
        self.sequence_mapping = copy.deepcopy(self.class_mapping)

    def _load_classifier(self, last_layer_name: str = 'fc', lateral_connections: Optional[List[str]] = None):
        """Load an already trained classifier model from synthetic data.

        Args:
            last_layer_name (str, optional): _description_. Defaults to 'fc'.
            lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # loading of dinoV2 representations
        try:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        except:
            dinov2 = torch.hub.load('/home_local/lee_jn/dinov2', 'dinov2_vits14_reg', source='local', pretrained=False)
            dinov2.load_state_dict(torch.load(self.args.model_path + '/dinov2/checkpoints/dinov2_vits14_reg4_pretrain.pth'))
            dinov2.to(self.device).eval()
            print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

        # loading of a classifier based on progressive neural netowrks
        mlp = ClassifierNet(input_size=list(dinov2.children())[-2].normalized_shape[0], output_size=2)
        classifier = copy.deepcopy(mlp).to(self.device)
        lateral_connections = None
        pnn = DropoutProgressiveNeuralNetwork(base_network=classifier,
                                              backbone=dinov2,
                                              last_layer_name=last_layer_name,
                                              lateral_connections=copy.deepcopy(lateral_connections),
                                              dropout_positions=['hidden_1', 'hidden_2'])
        
        # loading of a prior check-point
        pnn.to(self.device)
        return pnn
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        # update the class mapping
        try:
            self.class_mapping.update({object_name: int(list(self.class_mapping.values())[-1]+1)})
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)
        except IndexError:
            # due to no use of first column
            self.class_mapping.update({object_name: 0}) 
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)+1

        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        self.metric, self.classifier = pnn_fit_the_unknowns(model=self.classifier,
                                                            dataloader=dataloader,
                                                            device=self.device,
                                                            use_validation_set=False,
                                                            num_epochs=1,
                                                            is_classification=[True for i in range(column_num)])
        print("Learning the unknowns", self.metric)
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            column_num (int): _description_
        """
        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        # update the classifier
        self.metric, self.classifier = pnn_fit_the_knowns(model=self.classifier,
                                                          dataloader=dataloader,
                                                          device=self.device,
                                                          column_num=column_num,
                                                          use_validation_set=False,
                                                          num_epochs=1,
                                                          is_classification=[True for i in range(column_num+1)])
        print("Learning the knowns", self.metric)


class DeepEnsemble(TheSystem):
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 device: str,
                 sam_gap: int = 1000,
                 num_seg: int = 4,
                 is_temporal: bool = False,
                 class_mapping: dict = {"apple": 0},
                 mc_samples: int = 3,
                 pairwise_coupling: str = "normalization"):
        super().__init__(
            args, device, sam_gap, num_seg,
            is_temporal, class_mapping, mc_samples, pairwise_coupling)
        self.class_mapping = {}
        self.sequence_mapping = copy.deepcopy(self.class_mapping)

    def _load_classifier(self, last_layer_name: str = 'fc', lateral_connections: Optional[List[str]] = None):
        """Load an already trained classifier model from synthetic data.

        Args:
            last_layer_name (str, optional): _description_. Defaults to 'fc'.
            lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # loading of dinoV2 representations
        try:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        except:
            dinov2 = torch.hub.load('/home_local/lee_jn/dinov2', 'dinov2_vits14_reg', source='local', pretrained=False)
            dinov2.load_state_dict(torch.load(self.args.model_path + '/dinov2/checkpoints/dinov2_vits14_reg4_pretrain.pth'))
            dinov2.to(self.device).eval()
            print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

        # loading of a classifier based on progressive neural netowrks
        mlp = ClassifierNet(input_size=list(dinov2.children())[-2].normalized_shape[0], output_size=2)
        classifier = copy.deepcopy(mlp).to(self.device)
        lateral_connections = None
        pnn = ProgressiveNeuralNetwork(base_network=classifier,
                                       backbone=dinov2,
                                       last_layer_name=last_layer_name,
                                       lateral_connections=copy.deepcopy(lateral_connections),
                                      )
        
        # loading of a prior check-point
        pnn.to(self.device)
        return [pnn for i in range(self.mc_samples)]
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        # update the class mapping
        try:
            self.class_mapping.update({object_name: int(list(self.class_mapping.values())[-1]+1)})
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)
        except IndexError:
            # due to no use of first column
            self.class_mapping.update({object_name: 0}) 
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)+1

        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        for i in range(self.mc_samples):
            self.metric, classifier = pnn_fit_the_unknowns(model=self.classifier[i],
                                                           dataloader=dataloader,
                                                           device=self.device,
                                                           use_validation_set=False,
                                                           num_epochs=1,
                                                           is_classification=[True for i in range(column_num)])
            self.classifier[i] = copy.deepcopy(classifier)
        print("Learning the unknowns", self.metric)
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            column_num (int): _description_
        """
        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        # update the classifier
        for i in range(self.mc_samples):
            self.metric, classifier = pnn_fit_the_knowns(model=self.classifier[i],
                                                        dataloader=dataloader,
                                                        device=self.device,
                                                        column_num=column_num,
                                                        use_validation_set=False,
                                                        num_epochs=1,
                                                        is_classification=[True for i in range(column_num+1)])
            self.classifier[i] = copy.deepcopy(classifier)
        print("Learning the knowns", self.metric)


class LaplaceApproximation(TheSystem):
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 device: str,
                 sam_gap: int = 1000,
                 num_seg: int = 4,
                 is_temporal: bool = False,
                 class_mapping: dict = {"apple": 0},
                 mc_samples: int = 10,
                 pairwise_coupling: str = "normalization"):
        super().__init__(
            args, device, sam_gap, num_seg,
            is_temporal, class_mapping, mc_samples, pairwise_coupling)
        self.is_prior = False
        self.class_mapping = {}
        self.sequence_mapping = copy.deepcopy(self.class_mapping)
        
    def _load_classifier(self, last_layer_name: str = 'fc', lateral_connections: Optional[List[str]] = None):
        """Load an already trained classifier model from synthetic data.

        Args:
            last_layer_name (str, optional): _description_. Defaults to 'fc'.
            lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # loading of dinoV2 representations
        try:
            dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
        except:
            dinov2 = torch.hub.load('/home_local/lee_jn/dinov2', 'dinov2_vits14_reg', source='local', pretrained=False)
            dinov2.load_state_dict(torch.load(self.args.model_path + '/dinov2/checkpoints/dinov2_vits14_reg4_pretrain.pth'))
            dinov2.to(self.device).eval()
            print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

        # loading of a classifier based on progressive neural netowrks
        mlp = ClassifierNet(input_size=list(dinov2.children())[-2].normalized_shape[0], output_size=2)
        classifier = copy.deepcopy(mlp).to(self.device)
        lateral_connections = None
        pnn = ProgressiveNeuralNetwork(base_network=classifier,
                                       backbone=dinov2,
                                       last_layer_name=last_layer_name,
                                       lateral_connections=copy.deepcopy(lateral_connections))
        
        # loading of a prior check-point
        pnn.to(self.device)
        return pnn
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        # update the class mapping
        try:
            self.class_mapping.update({object_name: int(list(self.class_mapping.values())[-1]+1)})
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)
        except IndexError:
            # due to no use of first column
            self.class_mapping.update({object_name: 0}) 
            self.sequence_mapping[object_name] = 0
            column_num = len(self.class_mapping)+1

        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        if self.is_prior is True:
            # fitting the bpnn classifier
            self.metric ,self.classifier = bpnn_fit_the_unknowns(
                model=self.classifier,
                dataloader=dataloader,
                pac_bayes=False,
                train_dataset=dataloader[0])
            print("Learning the unknowns", self.metric)
        else:
            self.learn_priors_from_scratch(dataloader)
            self.is_prior = True
            self.metric = {'train': [{'accuracy': 50.0, 'loss': 0.0}], 'test': []}
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            column_num (int): _description_
        """
        if self.is_prior is True:
            # create the dataloader
            if dataloader is None:
                dataloader = self._load_dataloader(object_name, random_selection)

            # update an existing column posterior
            self.metric, self.classifier = bpnn_fit_the_knowns(
                model=self.classifier, 
                dataloader=dataloader,
                pac_bayes=False,
                train_dataset=dataloader[0],
                column_num=column_num)
            print("Learning the knowns:", self.metric)
        else:
            self.learn_priors_from_scratch(dataloader)
            self.is_prior = True
            self.metric = {'train': [{'accuracy': 50.0, 'loss': 0.0}], 'test': []}

    def learn_priors_from_scratch(self, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """[summary]

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader], optional): [description]. Defaults to None.
        """
        # train the model in a deterministic way
        metric, classifier = pnn_fit_the_unknowns(model=self.classifier,
                                                  dataloader=dataloader,
                                                  device=self.device,
                                                  use_validation_set=True,
                                                  num_epochs=1,
                                                  is_classification=[True for i in range(1)])

        # then use the model and get a prior
        output_dim=2
        dataloader = (output_dim, dataloader, torch.nn.CrossEntropyLoss())
        dataloader = [dataloader, dataloader]

        # get the prior
        priors = list()
        scaling_method = McAllesterScaling(confidence=0.8, fixed=(None, None, None), shared=(None, None, None))
        for n_column in range(len(dataloader)-1):
            prior = init_prior_per_column(base_network=self.classifier, 
                                          curvature_type=KFOC,
                                          isotropic_prior=True,
                                          dataloader=dataloader[n_column+1][1][0],
                                          weight_decay=1e-5,
                                          curvature_scaling=scaling_method,
                                          column_num=n_column)
            priors.append(prior)

        # declare the model
        bpnn = TheCleverNetwork(model=self.classifier, priors=priors, backbone=self.classifier.backbone, weight_decay=1e-5, pac_bayes_iterations=1)
        
        # assign the values
        self.classifier = copy.deepcopy(bpnn)


class CleverWithoutPacBayes(TheSystem):
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 device: str,
                 sam_gap: int = 1000,
                 num_seg: int = 4,
                 is_temporal: bool = False,
                 class_mapping: dict = {"apple": 0},
                 mc_samples: int = 30,
                 pairwise_coupling: str = "normalization"):
        super().__init__(
            args, device, sam_gap, num_seg,
            is_temporal, class_mapping, mc_samples, pairwise_coupling)
        
    def _load_classifier(self):
        """Load an already trained classifier model from synthetic data.

        Args:
            last_layer_name (str, optional): _description_. Defaults to 'fc'.
            lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        # initialize the model class without loading checkpoints
        pnn = clever_pnn(self.args, self.device)

        # declare the model (disabling pac bayes iterations to 1)
        bpnn = TheCleverNetwork(model=pnn, priors=None, backbone=pnn.backbone, weight_decay=1e-5, pac_bayes_iterations=1)
        if self.args.prior_pt_file is not None:
            bpnn.load_full_state_dict(torch.load(self.args.prior_pt_file))
        bpnn.to(self.device)
        return bpnn
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        # update the class mapping
        self.class_mapping.update({object_name: int(list(self.class_mapping.values())[-1]+1)})
        self.sequence_mapping[object_name] = 0

        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        # fitting the bpnn classifier
        self.metric ,self.classifier = bpnn_fit_the_unknowns(
            model=self.classifier,
            dataloader=dataloader,
            train_dataset=dataloader[0])
        print("Learning the unknowns", self.metric)
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            column_num (int): _description_
        """
        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        # update an existing column posterior
        self.metric, self.classifier = bpnn_fit_the_knowns(
            model=self.classifier, 
            dataloader=dataloader,
            train_dataset=dataloader[0],
            column_num=column_num)
        print("Learning the knowns:", self.metric)
