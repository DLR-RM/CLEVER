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
1. Increase the number of heads and calculate the memory and inference time.
2. Compare with purely deterministic setting.

Expectation: 
More memory and inference time than purely deterministic setting.
Also, just indicate how many objects one can handle with the chosen architectures.
Run-time can be briefly mentioned if not interesting due to implementation issues.
"""

import torch
import time
import copy
import argparse
import json
import torch.nn.functional as F
import torchvision.transforms as transforms

from PIL import Image
from fvcore.nn import FlopCountAnalysis
from clever.clever import TheMostCleverSystem, TheCleverSystem

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"

# TODO:
# 2. vary the size of the image (today)
# 3. include dirichlet model
# 4. include time update with filtering


def runtime(args):
    # set the device if cuda is available
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # initiate the clever system
    if args.mode=="clever":
        clever = TheMostCleverSystem(args=args, device=device)
    elif args.mode=="standard":
        clever = TheCleverSystem(args=args, device=device)
    else:
        raise AttributeError

    # load an image for testing
    image = Image.open('/home_local/lee_jn/CLEVER/data/images/picture1.jpg')
    transform = transforms.Compose([
                        transforms.ToTensor(),
                        transforms.Resize((args.image_size, args.image_size)),
                        transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                        ])
    inputs = transform(image)
    infer_batch = torch.stack([inputs]).to(device)

    # running the model for the first time
    output = clever.classifier(infer_batch)

    # computing the time to run the model
    runtime = []
    for i in range(0, 1000):
        start = time.time()
        if args.mode=="clever":
            output = clever.classifier.bayesian_analysis(infer_batch, num_samples=30)
        elif args.mode=="standard":
            output = clever.classifier(infer_batch)
        else:
            raise AttributeError
        end = time.time()
        runtime.append(end-start)
        print(i, "runtime:", end-start)

    # save the latest one
    filename = args.results_path + "1_runtime_" + args.mode + "_" + str(args.image_size) + ".json"
    with open(filename, "w") as file:
        json.dump(runtime, file)


if __name__ == "__main__":
    parser = argparse.ArgumentParser("We examine how many objects our architecture can handle")
    parser.add_argument("--model_path", help="Path to the trained model", required=True)
    parser.add_argument("--model_pt_file", type=str, default=None, help="Path to the model pt file")
    parser.add_argument("--prior_pt_file", type=str, default=None, help="Path to the prior pt file")
    parser.add_argument("--pooldata_dir", help="Path to the dataset location", required=True)
    parser.add_argument("--results_path", help="Path to the results", required=True)
    parser.add_argument("--mode", help="clever", required=True)
    parser.add_argument("--image_size", type=int, default=280, help="clever")
    args = parser.parse_args()

    runtime(args)