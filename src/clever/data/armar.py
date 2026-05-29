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
Implementation of Armar dataloader for active learning.
Dataloader here is for active learning online.
One difference is that we assume no masks, but pure classification.
"""

__author__ = "Jongseok Lee"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Jongseok Lee"
__email__ = "jongseok.lee@dlr.de"

import random
import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, Any, Dict, List


class ArmarDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 category: str,
                 sequence: int = None,
                 random_selection: bool = False,
                 split: str = 'train',
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None):
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.categories = [category]

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        # NOTE: iterate over the folders to get different masks.
        # get image data as well as target here.
        # make them balanced by randomly picking them.
        self.category_images = []
        self.category_targets = []
        self.balancing_images = []
        self.balancing_target = []

        if sequence is None:
            category_path = os.path.join(self.root, category)
            list_of_sequences = os.listdir(category_path)
        else:
            category_path = os.path.join(self.root, category)
            list_of_sequences = ['seq' + str(sequence)]

        # puts category
        for seq in list_of_sequences:
            category_image_path = os.path.join(category_path, seq)
            img_list = os.listdir(category_image_path)
            if random_selection:
                img_list = random.choices(img_list, k=int(len(img_list)*0.3))
            [self.category_images.append((category, os.path.join(category_image_path, img_name))) for img_name in img_list]
            [self.category_targets.append(1) for img_name in img_list]

        list_of_objects = os.listdir(self.root)
        for objects in list_of_objects:
            if objects != category:
                antiobject_path = os.path.join(self.root, objects)
                list_of_antiobject_sequences = os.listdir(antiobject_path)
                for seq in list_of_antiobject_sequences:
                    antiobject_image_path = os.path.join(antiobject_path, seq)
                    img_list = os.listdir(antiobject_image_path)
                    if random_selection:
                        img_list = random.choices(img_list, k=int(len(img_list)*0.3))
                    [self.balancing_images.append(('others', os.path.join(antiobject_image_path, img_name))) for img_name in img_list]
                    [self.balancing_target.append(0) for img_name in img_list]

        # training and validation splits splits
        balancing_idx = random.sample([i for i in range(len(self.balancing_images))], int(len(self.category_images)*0.8))
        category_idx = random.sample([i for i in range(len(self.category_images))], int(len(self.category_images)*0.8))

        # validation splits
        balancing_idx_val = list(set([i for i in range(len(self.balancing_images))])^set(list(balancing_idx)))
        balancing_idx_val = random.sample(balancing_idx_val, int(len(self.category_images)*0.2))
        category_idx_val = list(set([i for i in range(len(self.category_images))])^set(list(category_idx)))

        if self.split=='train':
            balancing_idx = np.random.choice(len(self.balancing_images), int(len(self.category_images)*0.8))
            category_idx = np.random.choice(len(self.category_images), int(len(self.category_images)*0.8))
            self.balancing_images = np.asarray(self.balancing_images)[balancing_idx]
            self.balancing_target = np.asarray(self.balancing_target)[balancing_idx]
            self.category_images = np.asarray(self.category_images)[category_idx]
            self.category_targets = np.asarray(self.category_targets)[category_idx]
            self.data = list(self.balancing_images) + list(self.category_images)
            self.targets =  list(self.balancing_target) + list(self.category_targets)
        elif self.split=='validation':
            self.balancing_images = np.asarray(self.balancing_images)[balancing_idx_val]
            self.balancing_target = np.asarray(self.balancing_target)[balancing_idx_val]
            self.category_images = np.asarray(self.category_images)[category_idx_val]
            self.category_targets = np.asarray(self.category_targets)[category_idx_val]
            self.data = list(self.balancing_images) + list(self.category_images)
            self.targets =  list(self.balancing_target) + list(self.category_targets)
        else:
            raise AttributeError

        if len(self.data) == 1:
            for i in range(4):
                # Duplicate element in self.data
                self.data.append(self.data[0])
            print("This shouldnt happen.")
            exit(0)
        
        self.classes = self.categories
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        target_name, img_name = data

        target = self.targets[index]
        target = np.array(target).reshape(-1).astype(np.int64).squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        """Returns the folder to the raw data."""
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def processed_folder(self) -> str:
        """Returns the folder to the processed data."""
        return os.path.join(self.root)

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.processed_folder, category)) for category in self.categories)
    
    def extra_repr(self) -> str:
        """Adds split to its representation."""
        return "Split: {}\nCategories: {}".format(self.split, self.categories)


class ArmarSequential(Dataset):
    def __init__(self, 
                 root: str, 
                 category: str,
                 sequence: List,
                 balancer: List,
                 object_names: List,
                 split: str = 'train',
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None):
        """Different to previous dataset is the utilization
        for the full-scale learning experiment from sequence 0
        to sequence 8, or more.

        Args:
            root (str): [description]
            category (str): [description]
            sequence (int, optional): [description]. Defaults to None.
            random_selection (bool, optional): [description]. Defaults to False.
            split (str, optional): [description]. Defaults to 'train'.
            transform (Optional[Callable], optional): [description]. Defaults to None.
            target_transform (Optional[Callable], optional): [description]. Defaults to None.

        Raises:
            RuntimeError: [description]
            AttributeError: [description]
        """
        self.root = root
        self.split = split
        self.transform = transform
        self.target_transform = target_transform
        self.categories = [category]

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')
        
        # NOTE: iterate over the folders to get different masks.
        # get image data as well as target here.
        # make them balanced by randomly picking them.
        self.category_images = []
        self.category_targets = []
        self.balancing_images = []
        self.balancing_target = []
        
        # categorial images
        category_path = os.path.join(self.root, category)
        list_of_sequences = ['seq' + str(sequence[-1])]
        for seq in list_of_sequences:
            category_image_path = os.path.join(category_path, seq)
            img_list = os.listdir(category_image_path)
            [self.category_images.append((category, os.path.join(category_image_path, img_name))) for img_name in img_list]
            [self.category_targets.append(1) for img_name in img_list]

        # balancing images (TODO)
        if len(sequence) == 1:
            for objects in object_names:
                if objects != category:
                    antiobject_path = os.path.join(self.root, objects)
                    for seq in sequence:
                        antiobject_image_path = os.path.join(antiobject_path, 'seq' + str(seq))
                        img_list = os.listdir(antiobject_image_path)
                        [self.balancing_images.append(('others', os.path.join(antiobject_image_path, img_name))) for img_name in img_list]
                        [self.balancing_target.append(0) for img_name in img_list]
        else:
            for objects in balancer:
                if objects != category:
                    antiobject_path = os.path.join(self.root, objects)
                    antiobject_image_path = os.path.join(antiobject_path, 'seq' + str(sequence[-1]))
                    img_list = os.listdir(antiobject_image_path)
                    [self.balancing_images.append(('others', os.path.join(antiobject_image_path, img_name))) for img_name in img_list]
                    [self.balancing_target.append(0) for img_name in img_list]
            for objects in object_names:
                if objects != category:
                    antiobject_path = os.path.join(self.root, objects)
                    for seq in sequence[:-1]:
                        antiobject_image_path = os.path.join(antiobject_path, 'seq' + str(seq))
                        img_list = os.listdir(antiobject_image_path)
                        [self.balancing_images.append(('others', os.path.join(antiobject_image_path, img_name))) for img_name in img_list]
                        [self.balancing_target.append(0) for img_name in img_list]

        # training and validation splits splits
        balancing_idx = random.sample([i for i in range(len(self.balancing_images))], int(len(self.category_images)*0.8))
        category_idx = random.sample([i for i in range(len(self.category_images))], int(len(self.category_images)*0.8))

        # validation splits
        balancing_idx_val = list(set([i for i in range(len(self.balancing_images))])^set(list(balancing_idx)))
        balancing_idx_val = random.sample(balancing_idx_val, int(len(self.category_images)*0.2))
        category_idx_val = list(set([i for i in range(len(self.category_images))])^set(list(category_idx)))

        if self.split=='train':
            balancing_idx = np.random.choice(len(self.balancing_images), int(len(self.category_images)*0.8))
            category_idx = np.random.choice(len(self.category_images), int(len(self.category_images)*0.8))
            self.balancing_images = np.asarray(self.balancing_images)[balancing_idx]
            self.balancing_target = np.asarray(self.balancing_target)[balancing_idx]
            self.category_images = np.asarray(self.category_images)[category_idx]
            self.category_targets = np.asarray(self.category_targets)[category_idx]
            self.data = list(self.balancing_images) + list(self.category_images)
            self.targets =  list(self.balancing_target) + list(self.category_targets)
        elif self.split=='validation':
            self.balancing_images = np.asarray(self.balancing_images)[balancing_idx_val]
            self.balancing_target = np.asarray(self.balancing_target)[balancing_idx_val]
            self.category_images = np.asarray(self.category_images)[category_idx_val]
            self.category_targets = np.asarray(self.category_targets)[category_idx_val]
            self.data = list(self.balancing_images) + list(self.category_images)
            self.targets =  list(self.balancing_target) + list(self.category_targets)
        else:
            raise AttributeError

        if len(self.data) == 1:
            for i in range(4):
                # Duplicate element in self.data
                self.data.append(self.data[0])
            print("This shouldnt happen.")
            exit(0)
        
        self.classes = self.categories
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        target_name, img_name = data

        target = self.targets[index]
        target = np.array(target).reshape(-1).astype(np.int64).squeeze()

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(img_name).convert('RGB')

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)

    @property
    def raw_folder(self) -> str:
        """Returns the folder to the raw data."""
        return os.path.join(self.root, self.__class__.__name__)

    @property
    def processed_folder(self) -> str:
        """Returns the folder to the processed data."""
        return os.path.join(self.root)

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.processed_folder, category)) for category in self.categories)
    
    def extra_repr(self) -> str:
        """Adds split to its representation."""
        return "Split: {}\nCategories: {}".format(self.split, self.categories)


def get_armar_dataloader(root: str,
                         objs: str,
                         data_transforms: Dict,
                         batch_size: int,
                         num_workers: int,
                         train: bool = True):
    datasets = []
    for obj in objs:
        datasets.append(ArmarDataset(root=root, 
                                     category=obj, 
                                     transform=data_transforms['train'] if train else data_transforms['val']))

    dataloaders = [DataLoader(dataset=dataset, 
                              batch_size=batch_size,
                              shuffle=True if train else False, 
                              num_workers=num_workers) for dataset in datasets]

    return dataloaders

def create_armar_dataset(output_crop: str,
                         new_category: str,
                         obj_id: int,
                         frame_idx: int,
                         frame: np.ndarray,
                         pred_mask: np.ndarray):
    # check if folder exists
    if not os.path.exists(os.path.join(output_crop, new_category)):
        # create new folder in cropped_objs
        os.mkdir(os.path.join(output_crop, new_category))

    # get crop from current frame
    crop_mask = (pred_mask == obj_id).astype(np.uint8)
    x,y,w,h = cv2.boundingRect(crop_mask)
    crop_obj = frame * (pred_mask == obj_id).astype(np.uint8)[...,None]
    crop_obj = crop_obj[y:y+h,x:x+w]
    crop_obj = Image.fromarray(cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB))
    crop_obj.save(os.path.join(output_crop, new_category, '{}_{}.png'.format(obj_id, frame_idx)))