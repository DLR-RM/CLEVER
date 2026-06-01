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
Implementation of HOWS Dataset for Pytorch.
Synthetic dataset will be used to acquire prior distribution.
Adapted from https://github.com/DLR-RM/RECALL
"""

import argparse

from clever.data.hows import get_paths

__author__ = "lee_jn"
__copyright__ = "lee_jn"
__license__ = "MIT"

if __name__ == "__main__":
    # arguments
    parser = argparse.ArgumentParser("Prepares subset of HOWS dataset for prior from simulation")
    parser.add_argument("--image_path", help="Path to the image location", required=True)
    parser.add_argument("--sequence_path", help="Path to the sequence location", required=True)
    parser.add_argument("--validation_percentage", help="Validation percentage", default=10)
    parser.add_argument("--seq_name", help="Name of the sequence", default="_clever_hows")
    args = parser.parse_args()

    # each objects have 6000 images.
    # some objects are deformable kitchen objects.
    # some objects are non-deformable objects.
    # idea is to form a prior; 
    seq_0 = ["apple"] # deformable
    seq_1 = ["banana"] # deformable
    seq_2 = ["bowl"] # non-deformable
    seq_3 = ["egg"] # deformable
    seq_4 = ["glass_bottle"] # non-deformable
    seq_5 = ["milk"] # deformable
    seq_6 = ["mug"] # non-deformable
    seq_7 = ["pear"] # deformable
    seq_8 = ["scissors"] # deformable
    seq_9 = ["bread"] # deformable
    seq_10 = ["can"] # deformable
    seq_11 = ["fork"] # non-deformable
    seq_12 = ["knife"] # non-deformable
    seq_13 = ["pan"] # non-deformable
    seq_14 = ["spoon"] # non-deformable
    seqs = [seq_0, seq_1, seq_2, seq_3, seq_4, seq_5, seq_6, \
        seq_7, seq_8, seq_9, seq_10, seq_11, seq_12, seq_13, seq_14]

    # save the images / path names of considered objects
    for i, seq in enumerate(seqs):
        get_paths(seq, i, args.image_path, args.sequence_path, args.seq_name, True, args.validation_percentage)
    print("Done")