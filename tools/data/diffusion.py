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
Implementation of stable diffusion for synthetic data generation.
Such data can be used for the prior learning, which I love very much.

See: https://huggingface.co/stable-diffusion-v1-5/stable-diffusion-v1-5/tree/main
"""

import argparse
import torch

from diffusers import DiffusionPipeline

def main(args):
    objects = ["apple", "bowl", "cleaning_cloth", "glass_bottle", \
        "kendo_head_cover", "keyboard", "knife", "milk", "pear", "spoon", "t_shirt"]
    prompts = dict()
    prompts[objects[0]] = "An apple on a white table."
    prompts[objects[1]] = "An empty plastic bowl on top of a white table."
    prompts[objects[2]] = "A yellow cleaning cloth on a white table."
    prompts[objects[3]] = "A water glass bottle on a white table."
    prompts[objects[4]] = "The Great Wave of Kanagawa Fabric Block" #"A small colorful piece of rectangular fabric on top of a white table."
    prompts[objects[5]] = "A black keyboard on a white table."
    prompts[objects[6]] = "A small butter knife, silver color, on top of a white table with no background."
    prompts[objects[7]] = "A wooden milk package on a white table."
    prompts[objects[8]] = "A pear on a white table."
    prompts[objects[9]] = "A small silver spoon on a white table."
    prompts[objects[10]] = "A rectangular gray t-shirt folded on a white table."

    stable_diffusion = DiffusionPipeline.from_pretrained(args.model_dir, use_safetensors=True)
    stable_diffusion = stable_diffusion.to("cuda")

    img_num = 2500
    for i in range(0,2500):
        for obj_cat in objects:
            # generating the files
            prompt = prompts[obj_cat]
            image = stable_diffusion(prompt).images[0]  

            # saving the image
            filename = args.output_dir + obj_cat + "/seq0/" + obj_cat + "_" + str(img_num) + ".jpg"
            image.save(filename)
        img_num = img_num + 1


if __name__ == "__main__":
    parser = argparse.ArgumentParser("Lets try to learn a prior distribution")
    parser.add_argument("--output_dir", type=str, default="/home_local/lee_jn/armar2/", help="Name of the object")
    parser.add_argument("--model_dir", type=str, default="/home_local/lee_jn/stable-diffusion-v1-5", help="Output directory")
    args = parser.parse_args()
    main(args)
