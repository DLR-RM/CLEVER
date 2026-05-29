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
Generator args for mobile sam, aot and combined tracker.
Modificaiton based on https://github.com/z-x-yang/Segment-and-Track-Anything
"""

# Explanation of generator_args is in sam/segment_anything/automatic_mask_generator.py: SamAutomaticMaskGenerator
sam_args = {
    'sam_checkpoint': "/home_local/lee_jn/CLEVER/data/model/mobilesam/weight/sam_vit_h_4b8939.pth", 
    #'sam_checkpoint': "/common/homes/external/dy6919_lee/Packages/clever/data/model/mobilesam/weight/sam_vit_h_4b8939.pth", 
    #'sam_checkpoint': "/home/lee/Packages/clever/data/model/mobilesam/weight/mobile_sam.pt",
    'model_type': "vit_h",
    'generator_args':{
        'points_per_side': 32,
        'pred_iou_thresh': 0.8,
        'stability_score_thresh': 0.9,
        'crop_n_layers': 1,
        'crop_n_points_downscale_factor': 2,
        'min_mask_region_area': 1500,
    },
    'gpu_id': 0,
}

aot_args = {
    'phase': 'PRE_YTB_DAV',
    'model': 'r50_deaotl',
    'model_path': '/home_local/lee_jn/CLEVER/data/model/aot/R50_DeAOTL_PRE_YTB_DAV.pth',
    #'model_path': '/common/homes/external/dy6919_lee/Packages/clever/data/model/aot/R50_DeAOTL_PRE_YTB_DAV.pth', #'/home/lee/Packages/clever/data/model/aot/R50_DeAOTL_PRE_YTB_DAV.pth',
    #'model_path': '/home/lee/Packages/clever/data/model/aot/R50_DeAOTL_PRE_YTB_DAV.pth',
    'long_term_mem_gap': 9999,
    'max_len_long_term': 9999,
    'gpu_id': 0,
}

segtracker_args = {
    'sam_gap': 9999, # the interval to run sam to segment new objects
    'min_area': 500, # minimal mask area to add a new mask as a new object
    'max_obj_num': 3, # maximal object number to track in a video
    'min_new_obj_iou': 0.8, # the background area ratio of a new object should > 80% 
}