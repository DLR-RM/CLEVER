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
Implementation of TORO Dataset.
"""

__author__ = "Jongseok Lee"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Jongseok Lee"
__email__ = "jongseok.lee@dlr.de"

import os
import cv2
import numpy as np

from PIL import Image
from torch.utils.data import Dataset, DataLoader
from typing import Optional, Callable, Tuple, Any, Dict, List


class FolderDataset(Dataset):
    def __init__(self, 
                 root: str, 
                 category: str,
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None):
        self.root = root
        self.transform = transform
        self.target_transform = target_transform
        self.categories = [category]

        if not self._check_exists():
            raise RuntimeError('Dataset not found.')

        self.data = []
        img_list = os.listdir(os.path.join(self.root, category))
        [self.data.append((category, img_name)) for img_name in img_list]

        if len(self.data) == 1:
            for i in range(4):
                # Duplicate element in self.data
                self.data.append(self.data[0])

        self.classes = self.categories
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        target_name, img_name = data
        target = self.categories.index(target_name)

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.open(os.path.join(self.processed_folder, *data)).convert('RGB')

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

def get_folder_dataloader(root: str,
                          objs: str,
                          data_transforms: Dict,
                          batch_size: int,
                          num_workers: int,
                          train: bool = True):
    datasets = []
    for obj in objs:
        datasets.append(FolderDataset(root=root, 
                                      category=obj, 
                                      transform=data_transforms['train'] if train else data_transforms['val']))

    dataloaders = [DataLoader(dataset=dataset, 
                              batch_size=batch_size,
                              shuffle=True if train else False, 
                              num_workers=num_workers) for dataset in datasets]

    return dataloaders

def create_folder_dataset(output_crop: str,
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


class ImageDataset(Dataset):
    def __init__(self, 
                 img: List[np.ndarray], 
                 category: List[str],
                 transform: Optional[Callable] = None, 
                 target_transform: Optional[Callable] = None):
        self.transform = transform
        self.target_transform = target_transform

        self.categories = category

        self.data = []
        [self.data.append((category, img_array)) for category, img_array in zip(category,img)] 

        self.classes = self.categories
    
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        target_name, img_array = data
        target = self.categories.index(target_name)

        img = Image.fromarray(img_array)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target

    def __len__(self) -> int:
        return len(self.data)


class TORODataset(Dataset):
    # Dict with all the categories and their appearing sequences
    TORO_cats_seqs = {'dark_mat':       [1,2,3,4,5,6,7],
                      'lab_floor':      [1,2,3,4,5,6,7], 
                      'stones':         [1,2,3,4,5,6,7],
                      'ramp':           [2,4,5,6,7], 
                      'blue_mattress':  [2,3,6,7], 
                      'pedestal':       [3,6,7],
                      'wall':           [1,2,3,4,7], 
                      'PC_table':       [2,4,7], 
                      'plane_wall':     [2,3,4],
                      'extinguisher':   [2,4,7], 
                      'cable':          [2,4,7], 
                      'TV_mount':       [1,2,3], 
                      'bistro_table':   [2,7], 
                      'person':         [5,6], 
                      'stand_board':    [2,4],}
    all_cats_seqs = TORO_cats_seqs

    def __init__(
            self,
            root: str,
            cat_seq: Optional[Dict[str, List[int]]] = None,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            max_data_seq: Optional[int] = 0,
            randomize: Optional[bool] = False,
    ) -> None:        
        self.root = root
        self.transform = transform
        self.target_transform = target_transform

        self.categories = [category for category in self.all_cats_seqs.keys()
                           if (cat_seq is None or category in cat_seq.keys())]
        
        self.sequences = [(category, str(sequence))
                          for category in self.categories
                          for sequence in self.all_cats_seqs[category]
                          if (cat_seq is None or sequence in cat_seq[category])]
        
        if not self._check_exists():
            raise RuntimeError('\x1b[1;37;41m' + 'Dataset not found' + '\x1b[0m')
        
        self.data = [] # [(category, sequence, img_name), ...]
        for category, sequence in self.sequences:
            seq_path = os.path.join(self.processed_folder, category, sequence)

            if randomize:
                if max_data_seq is not 0:
                    max_len = max_data_seq if max_data_seq < len(os.listdir(seq_path)) else len(os.listdir(seq_path))
                else:
                    max_len = len(os.listdir(seq_path))

                random_img_list = np.random.choice(os.listdir(seq_path), max_len, replace=False)
                [self.data.append((category, sequence, img_name)) for img_name in random_img_list]
            else:
                img_list = os.listdir(seq_path)

                if max_data_seq is not 0:
                    [self.data.append((category, sequence, img_name)) for img_name in img_list[:max_data_seq]]
                else:
                    [self.data.append((category, sequence, img_name)) for img_name in img_list]

        self.classes = self.categories

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        data = self.data[index]
        category_name, sequence_name, img_name = data
        target = self.categories.index(category_name)

        # return a PIL Image
        img = Image.open(os.path.join(self.processed_folder, *data)).convert('RGB')

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
        return os.path.join(self.root) #, self.__class__.__name__), 'rgbd-dataset')

    @property
    def class_to_idx(self) -> Dict[str, int]:
        """Returns the dict that maps the class to its index."""
        return {_class: i for i, _class in enumerate(self.classes)}
    
    @property
    def num_classes(self) -> int:
        """Returns the number of classes."""
        return len(self.classes)

    def _check_exists(self) -> bool:
        return all(os.path.exists(os.path.join(self.processed_folder, category)) for category in self.categories)

    def extra_repr(self) -> str:
        """Adds split to its representation."""
        return "Split: {}\nCategories: {}".format(self.split, self.categories)
        
def get_split_TOROdataset(dataset_path: str,
                          tasks: List[str],
                          train_ratio: float,
                          data_transforms: Dict[str, Callable] = None,
                          batch_size: int = 1,
                          num_workers: int = 0,
                          exclude_seqs: List[int] = None,
                          max_data_seq: Optional[int] = 0,
                          randomize: Optional[bool] = False):
    # Copy TORO_cats_seqs to TORO_cats_seqs_copy
    TORO_cats_seqs_copy = {task: TORODataset.all_cats_seqs[task].copy() for task in tasks}

    if exclude_seqs is not None:
        for task in tasks:
            TORO_cats_seqs_copy[task] = [seq for seq in TORO_cats_seqs_copy[task] if seq not in exclude_seqs]

    N_seq = np.array([len(TORO_cats_seqs_copy[task]) for task in tasks]) 
    N_seq_train = np.floor(N_seq * train_ratio)
    train_datasets = []
    val_datasets = []
    test_datasets = []
    for i, task in enumerate(tasks):
        random_seq = np.random.choice(TORO_cats_seqs_copy[task], N_seq[i], replace=False)
        random_train_seq = random_seq[:int(N_seq_train[i])]
        random_test_seq = random_seq[int(N_seq_train[i]):]
        train_datasets.append(TORODataset(root = dataset_path,
                                          cat_seq = {task: random_train_seq},
                                          transform = data_transforms['train'] if data_transforms is not None else None,
                                          max_data_seq = max_data_seq,
                                          randomize = randomize))
        val_datasets.append(TORODataset(root = dataset_path,
                                        cat_seq = {task: random_train_seq},
                                        transform = data_transforms['val'] if data_transforms is not None else None,
                                        max_data_seq = int(np.ceil(max_data_seq*0.3) if max_data_seq is not 0 else 0),
                                        randomize = randomize))
        test_datasets.append(TORODataset(root = dataset_path,
                                         cat_seq = {task: random_test_seq},
                                         transform = data_transforms['test'] if data_transforms is not None else None,
                                         max_data_seq = max_data_seq,
                                         randomize = randomize))

    train_dataloaders = [DataLoader(train_dataset,
                                    batch_size=batch_size,
                                    shuffle=True,
                                    num_workers=num_workers) for train_dataset in train_datasets]
    val_dataloaders = [DataLoader(val_dataset,
                                  batch_size=batch_size,
                                  shuffle=False,
                                  num_workers=num_workers) for val_dataset in val_datasets]
    test_dataloaders = [DataLoader(test_dataset,
                                   batch_size=batch_size,
                                   shuffle=False,
                                   num_workers=num_workers) for test_dataset in test_datasets]
        
    return train_dataloaders, val_dataloaders, test_dataloaders

