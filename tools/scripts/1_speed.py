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
Vary number of data points and number of parameters, then plot for the training time.

Expectation: 
Speed Vs number of parameters.
Speed Vs number of data points.

5 x 5 = 25 runs OR
10 x 10 = 10 runs

We can plot a grid for the given example OR plot a bar chart with high and lows.
"""
import copy
import argparse
import torch
import json
import time

from torch import nn, Tensor
from torch.nn import functional as F
from torchvision import transforms
from bpnn.pnn import ProgressiveNeuralNetwork
from clever.data.clever import CleverDataset
from clever.activelearner import ActiveLearningData
from clever.trainer import pnn_fit_the_knowns, pnn_fit_the_unknowns

def get_data(args, data_size:int):
    # initialization
    batch_size = 4
    training_iterations = 4096 * 6
    random_selection = False

    # define dataset
    transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Resize((280, 280)),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                ])
    train_dataset = CleverDataset(root=args.pooldata_dir, 
                                  split='train', 
                                  category="cigarette", 
                                  transform=transform, 
                                  random_selection=random_selection,
                                  sequence=str(0))
    
    # reduce training data
    al_data = ActiveLearningData(train_dataset)
    initial_samples = [*range(0, data_size, 1)] 
    al_data.acquire(initial_samples)

    # return dataloader
    train_loader = torch.utils.data.DataLoader(
    al_data.training_dataset,
    batch_size=batch_size,
    )
    return train_loader

class ClassifierNet(nn.Module):
    def __init__(self, input_size: int, output_size: int, layer_num: int):
        super().__init__()
        self.input = nn.Linear(in_features=input_size, out_features=288)
        self.layers = nn.ModuleList()
        for i in range(layer_num):
            self.layers.append(nn.Linear(in_features=288, out_features=288))
        self.fc = nn.Linear(in_features=288, out_features=output_size)
        
    def forward(self, x):
        x = F.relu(self.input(x))
        # x = F.relu(self.hidden_1(x))
        # x = F.relu(self.hidden_2(x))
        for layer in self.layers:
            x = F.relu(layer(x))
        return self.fc(x)

def get_model(args, device, layer_num):
    try:
        dinov2 = torch.hub.load('facebookresearch/dinov2', 'dinov2_vits14_reg')
    except:
        dinov2 = torch.hub.load('dinov2', 'dinov2_vits14_reg', source='local', pretrained=False)
        dinov2.load_state_dict(torch.load(args.model_path + '/dinov2/checkpoints/dinov2_vits14_reg4_pretrain.pth'))
        dinov2.to(device).eval()
        print('\x1b[1;37;42m' + '>> DINOv2 model loaded locally' + '\x1b[0m')

    # loading of a classifier based on progressive neural netowrks
    mlp = ClassifierNet(input_size=list(dinov2.children())[-2].normalized_shape[0], output_size=2, layer_num=layer_num)
    classifier = copy.deepcopy(mlp).to(device)
    lateral_connections = None
    pnn = ProgressiveNeuralNetwork(base_network=classifier,
                                   backbone=dinov2,
                                   last_layer_name='fc',
                                   lateral_connections=copy.deepcopy(lateral_connections))

    pnn.to(device)
    return pnn

def main(args):
    # initialization
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    column_num = 1

    # define simple dataset (try to use the existing set of data)
    dataloader = get_data(args, args.data_size)
    dataloader = (dataloader, None, None)

    # define simple mlp (see where you define it and make it as a function of layers)
    model = get_model(args, device, args.model_size)
    total_params = sum(p.numel() for p in model.parameters())

    # meausre training time
    trainingtime = []
    for i in range(0, 100):
        start_time = time.time()
        metric, classifier = pnn_fit_the_unknowns(model=model,
                                                 dataloader=dataloader,
                                                 device=device,
                                                 use_validation_set=False,
                                                 num_epochs=1,
                                                 is_classification=[True for i in range(column_num)])
        end_time = time.time()
        print("training time", end_time-start_time)
        trainingtime.append(end_time-start_time)

    # save the latest one
    filename = args.results_path + "1_speed_" + str(args.data_size) + "_" + str(args.model_size) + "_" + str(total_params) + ".json"
    with open(filename, "w") as file:
        json.dump(trainingtime, file)

if __name__ == "__main__":
    parser = argparse.ArgumentParser("We examine how many objects our architecture can handle")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--results_path", help="Path to the results", required=True)
    parser.add_argument("--data_size", type=int, default=100, help="clever")
    parser.add_argument("--model_size", type=int, default=2, help="clever")
    args = parser.parse_args()
    main(args)