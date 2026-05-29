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
Adapted from https://github.com/DLR-RM/RECALL
"""

__author__ = "Jongseok Lee"
__license__ = "MIT"
__version__ = "1.0"
__maintainer__ = "Jongseok Lee"
__email__ = "jongseok.lee@dlr.de"

import os
import h5py
import torch
import numpy as np
import argparse

from os.path import dirname, join
from pathlib import Path
from typing import Optional, Callable, Tuple, Any, Dict, List
from PIL import Image
from torchvision import transforms
from torch.utils.data import Dataset, DataLoader
from torchvision.datasets import VisionDataset

# TODO: write the docstring
# TODO: modify so that you only load one sets of classes upon a call (assign none to an image; then when being called, load them.).


class HOWSImport(object):
    def __init__(self, root_file: str, class_mapping: dict = None):
        """Get pictures of a specific sequence and training or validation.

        Args:
            root_file (str): filename of the text file where the picture-paths are in. For example, training_0.txt.
            class_mapping (dict, optional): mapping of the class ID. Defaults to None.
        """
        self.images = np.array([])
        self.labels = []
        self.nr_classes = None
        self.nr_images = None
        self.paths = []
        if class_mapping is None:
            self.class_mapping = {
                "apple": 0, "ball": 1, "bowl": 2, "camera": 3, "cap": 4, "egg": 5, "glass_bottle": 6, "headset": 7,
                "milk": 8, "mug": 9, "pear": 10, "scissors": 11, "teddy": 12, "bag": 13, "banana": 14, "bread": 15, "can": 16,
                "computer_keyboard": 17, "fork": 18, "glasses": 19, "knife": 20, "mobile_phone": 21, "pan": 22, "pen": 23, "spoon": 24
            }
            exit(0)
        else:
            self.class_mapping = class_mapping

        with open(root_file, "r") as f:
            self.paths.extend(f.read().split("\n"))

        self.paths = list(filter(None, self.paths))  # checks if there is a empty string in the list and removes it
        self.nr_images = len(self.paths)
        self.images = [self.load_image(join(str(Path(root_file).parent.parent), filename)) for filename in self.paths]

        class_names = []
        for filename in self.paths:
            name = os.path.basename(dirname(dirname(dirname(filename))))
            if name == "teddy_fix":
                class_names.append("teddy")
            else:
                class_names.append(name)

        for obj in class_names:
            for key, value in self.class_mapping.items():
                if obj.startswith(key):
                    self.labels.append(value)
                    break

        self.images = np.array(self.images)
        self.labels = np.array(self.labels).reshape(-1).astype(np.int64)
        self.paths = np.array(self.paths)
        self.nr_classes = len(class_mapping)
        self.labels = np.eye(self.nr_classes)[self.labels].astype(np.float32) 

    def load_memory(self):
        return None

    def get_X(self, max_amount=None):
        if max_amount is None:
            return self.images
        else:
            return self.images[:max_amount]

    def get_Y(self, max_amount=None):
        if max_amount is None:
            return self.labels
        else:
            return self.labels[:max_amount]

    def get_paths(self, max_amount=None):
        if max_amount is None:
            return self.paths
        else:
            return self.paths[:max_amount]

    def load_image(self, path: str) -> np.array:
        """Loads image from a given .hdf5 container path

        Args:
            path (str): path to .hdf5 container

        Raises:
            Exception: Path does not exist.

        Returns:
            RGB Image in numpy format.
        """
        if os.path.exists(path):
            if os.path.isfile(path):
                if path.endswith(".hdf5"):
                    with h5py.File(path, 'r') as data:
                        img = np.array(data["colors"], dtype=np.float32)
                    return img
        else:
            raise Exception(f"Path {path} doesnt exist!")

def write_file(data: List[str], sequence: str, mode: str, sequence_path: str, seq_name: str = "default"):
    """Writes the image order into a .txt file.

    Args:
        data (List[str]): list of paths to HOWS images.
        sequence (str): name of the sequence.
        mode (str): validation or training.
        sequence_path (str): path to dataset sequences, ex) /HOWS_CL_25/Sequences.
        seq_name (str, optional): sequence name. Defaults to "default".
    """
    if mode == "training":
        file_name = "training_{}{}.txt".format(sequence, seq_name)
    elif mode == "validation":
        file_name = "validation_{}{}.txt".format(sequence, seq_name)
    else:
        raise Exception("Mode {} not supported yet".format(mode))

    output_path = sequence_path + file_name

    with open(output_path, 'w') as f:
        f.write("\n".join(data))

def get_paths(sequence: List[str], 
              nr: int, 
              image_path: str, 
              sequence_path: str, 
              seq_name: str = "default",
              writing_mode: bool = True, 
              validation_percentage: float = 10):
    """Gets the paths of images.

    Args:
        sequence (List[str]): list of categories in the sequence.
        nr (int): sequence number.
        image_path (str): path to images, ex) /HOWS_CL_25/Images
        sequence_path (str): path to dataset sequences, ex) /HOWS_CL_25/Sequences.
        seq_name (str, optional): _sequence name. Defaults to "default".
        writing_mode (bool, optional): ff true the paths are written into the file, if not not. Defaults to True.
        validation_percentage (float, optional): the first 10 % Instance of each object is for validation, each else for training. Defaults to 10.
    """
    validation = []
    training = []
    repeat = True

    for obj in sequence:
        obj_path = join(image_path, obj)
        count_files = len(os.listdir(obj_path))
        count_valid_files = 0
        lst = os.listdir(obj_path)
        lst.sort()
        for instance_f in lst:
            instance_path = join(obj_path, instance_f)
            repeat = (count_valid_files / float(count_files) * 100) < validation_percentage
            for hash_file in os.listdir(instance_path):
                file_path = os.path.join(instance_path, hash_file)
                for picture in os.listdir(file_path):
                    if picture.endswith(".hdf5"):
                        picture_path = join(file_path, picture)
                        if os.path.exists(picture_path):
                            if repeat:
                                validation.append(os.path.join(file_path, picture))
                            else:
                                training.append(os.path.join(file_path, picture))
                        else:
                            raise FileNotFoundError("File {} does not exist!".format(picture_path))
            if repeat:
                count_valid_files += 1

        if writing_mode:
            write_file(training, nr, "training", sequence_path, seq_name)
            write_file(validation, nr, "validation", sequence_path, seq_name)

    return training, validation


class HOWS(VisionDataset):
    def __init__(
            self,
            root: str,
            images: np.array,
            labels: np.array,
            transform: Optional[Callable] = None,
            target_transform: Optional[Callable] = None,
            device: str = 'cpu',
    ) -> None:
        """HOWS dataset

        Args:
            root (str): Path to the dataset location
            images (np.array): Array of images
            labels (np.array): Array of lables
            transform (Optional[Callable], optional): torchvision.transforms. Defaults to None.
            target_transform (Optional[Callable], optional): torchvision.transforms. Defaults to None.
            device (str, optional): Device to be used (cuda or cpu). Defaults to 'cpu'.

        Raises:
            RuntimeError: Dataset not found on root folder.
        """
        super().__init__(root, transform=transform, target_transform=target_transform)

        if not os.path.exists(self.root):
            raise RuntimeError(
                'Dataset not found.' +
                ' You can use download=True to download it')
        self.device = device
        self.images = torch.from_numpy(images).permute(0, 3, 1, 2)
        self.labels = np.argmax(labels, axis=1)
        self.labels = torch.from_numpy(self.labels) # TODO: GPU util optimization

    def _preprocess(self, image):
            return image
        
    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        # index
        img = self.images[index]
        target = self.labels[index]

        # transforms
        if self.transform is not None:
            img = self.transform(img)
        if self.target_transform is not None:
            target = self.target_transform(target)

        return img.to(self.device), target.to(self.device)

    def __len__(self) -> int:
        return self.images.shape[0]

def class_balancing_data(final_path: str, 
                         sequence: str, 
                         class_mapping: dict, 
                         data_size: int, 
                         mode: str, 
                         seq_name: str = '_clever_hows', 
                         nr_classes: int = 2):
    """_summary_

    Args:
        final_path (str): _description_
        sequence (str): _description_
        class_mapping (dict): _description_
        data_size (int): _description_
        mode (str): Mode of the dataset (train, validation).
        seq_name (str, optional): _description_. Defaults to '_clever_hows'.
        nr_classes (int, optional): _description_. Defaults to 15.

    Returns:
        _type_: _description_
    """
    # list possible variations of txt files
    sequences = [i for i in range(nr_classes) if i not in [int(sequence)]]

    # train or validation
    if mode == "train":
        final_paths = [final_path + "/training_{}{}.txt".format(sequence, seq_name) for sequence in sequences]
    elif mode == "validation":
        final_paths = [final_path + "/validation_{}{}.txt".format(sequence, seq_name) for sequence in sequences]
    else:
        raise Exception("This combination does not exist!")

    # obtain data of the specific size
    target_nr = int(data_size/(nr_classes-1))
    images_list = []
    for final_path in final_paths:
        image_provider = HOWSImport(final_path, class_mapping=class_mapping)
        images = image_provider.get_X() 
        idx = np.random.choice(images.shape[0], target_nr)
        images_list.append(images[idx, :, :, :])
    extra_images=np.concatenate(images_list, axis=0)
    labels = np.zeros((extra_images.shape[0], )).astype(np.int64)
    extra_labels=np.eye(2)[labels].astype(np.float32) # TODO: more general than typing 2?
    return extra_images , extra_labels 

def HOWS_CL_25(root: str, 
               mode: str, 
               batch_size: int = 16, 
               sequence: str = "", 
               num_workers: int = 8,
               transform: Optional[Callable] = None,
               target_transform: Optional[Callable] = None,
               device: str = 'cpu',
               seq_name: str = '_clever_hows',
               class_mapping: dict = None
              ):
    """_summary_

    Args:
        root (str): Path to the dataset location.
        mode (str): Mode of the dataset (train, validation).
        batch_size (int, optional): Size of the batch. Defaults to 16.
        sequence (str, optional): Category in the sequence. Defaults to "".
        num_workers (int, optional): Number of workers. Defaults to 8.
        transform (Optional[Callable], optional): torchvision.transforms. Defaults to None.
        target_transform (Optional[Callable], optional): torchvision.transforms. Defaults to None.
        device (str, optional): Device to be used (cuda or cpu). Defaults to 'cpu'.
        seq_name (str, optional): _sequence name. Defaults to "default".

    Returns:
        _type_: _description_
    """
    # path to sequences of HOWS_CL_25
    final_path = os.path.join(root, "HOWS_CL_25", "Sequences")

    # train or validation
    if mode == "train":
        _final_path = final_path + "/training_{}{}.txt".format(sequence, seq_name)
        shuffle = True
    elif mode == "validation":
        _final_path = final_path + "/validation_{}{}.txt".format(sequence, seq_name)
        shuffle = False
    else:
        raise Exception("This combination does not exist!")
    
    # load the images and labels
    if os.path.exists(_final_path):
        image_provider = HOWSImport(_final_path, class_mapping=class_mapping)
    else:
        raise Exception("The created path does not exist: {}".format(_final_path))
    images, labels = image_provider.get_X(), image_provider.get_Y()

    # get images to balance with other classes
    extra_images , extra_labels = class_balancing_data(final_path=final_path, 
                                                       sequence=sequence, 
                                                       class_mapping=class_mapping, 
                                                       data_size=images.shape[0],
                                                       mode=mode)
    
    # combining to make a balanced image
    final_images = np.concatenate((images, extra_images), axis=0)
    final_labels = np.concatenate((labels, extra_labels), axis=0)

    # return torch dataloder class
    HowsDataset = HOWS(root=_final_path, 
                       images=final_images, 
                       labels=final_labels,
                       transform=transform,
                       target_transform=target_transform,
                       device=device
                       )
    
    # return dataloader class
    dataloader = DataLoader(HowsDataset,
                            batch_size=batch_size,
                            shuffle=shuffle,
                            num_workers=num_workers)
    return dataloader

def get_train_val_test_split_dataloaders(
        dataset_class: type,
        data_path: str,
        splits: List,
        download: bool,
        transform,
        batch_size: int = 1,
        kwargs: Optional[Dict] = None) \
        -> Tuple[int, Tuple[DataLoader, ...], torch.nn.Module]:
    """Returns the training, validation and test split for a dataset.

    Args:
        dataset_class: A dataset class
        data_path: The path to the datasets
        splits: A list of [training, validation, test] for the split
            parameter of the dataset class
        download: Whether the datasets should be downloaded
        transform: A torchvision transform
        batch_size: The batch size for the dataloader
        kwargs: Keyword arguments for the dataset class

    Returns:
        The number of classes in the classification problem and the dataloaders
    """
    if kwargs is None:
        kwargs = {}
    seed = 0
    generator = torch.Generator().manual_seed(seed)
    datasets = [dataset_class(root=data_path, split=split, download=download, transform=transform, **kwargs) for split
                in
                splits]
    dataloaders = [DataLoader(
        dataset, pin_memory=torch.cuda.is_available(), num_workers=len(os.sched_getaffinity(0)),
        batch_size=batch_size, shuffle=(i == 0), generator=generator)
                   for i, dataset in enumerate(datasets)]
    output_size = len(datasets[0].classes)
    return output_size, tuple(dataloaders), torch.nn.CrossEntropyLoss()

def get_pnn_dataloaders(args: argparse.ArgumentParser):
    """_summary_

    Args:
        args (argparse.ArgumentParser): _description_

    Returns:
        _type_: _description_
    """
    # define data transforms 
    transform = transforms.Compose([
                transforms.Resize(280),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    # define class mapping
    # class_mapping = {
    #     "apple": 0, "banana": 1, "bowl": 2, "egg": 3, "glass_bottle": 4, "milk": 5, "mug": 6, "pear": 7,
    #     "scissors": 8, "bread": 9, "can": 10, "fork": 11, "knife": 12, "pan": 13, "spoon": 14
    # }
    class_mapping = {
        "apple": 0, "banana": 1
    }
    output_dim = 2

    # initialize
    dataloaders = []
    sequences = [i for i in range(0, len(class_mapping))]
    sequences.insert(0, 0) # NOTE: prior should not count for pnn
    for seq in sequences:
        mapping = multiclass_binaryclass(class_mapping, seq)
        train_data = HOWS_CL_25(root=args.root,
                                mode="train", 
                                batch_size=args.batch_size, 
                                sequence=seq, 
                                num_workers=8,
                                transform=transform,
                                class_mapping=mapping)
        test_data = HOWS_CL_25(root=args.root,
                               mode="validation", 
                               batch_size=args.batch_size, 
                               sequence=seq, 
                               num_workers=8,
                               transform=transform,
                               class_mapping=mapping)
        dataloaders.append((output_dim, (train_data, test_data, test_data), torch.nn.CrossEntropyLoss()))
    return dataloaders

def get_pnn_data_tuple():
    dataloaders = []
    # class_mapping = {
    #     "apple": 0, "banana": 1, "bowl": 2, "egg": 3, "glass_bottle": 4, "milk": 5, "mug": 6, "pear": 7,
    #     "scissors": 8, "bread": 9, "can": 10, "fork": 11, "knife": 12, "pan": 13, "spoon": 14
    # }
    class_mapping = {
        "apple": 0, "banana": 1
    }
    output_dim = 2
    sequences = [i for i in range(0, len(class_mapping))]
    # sequences = [1]
    sequences.insert(0, 0) # NOTE: prior should not count for pnn
    for seq in sequences:
        dataloaders.append((output_dim, str(seq), torch.nn.CrossEntropyLoss()))
    return dataloaders

def multiclass_binaryclass(class_mapping: dict, column_num: int):
    """_summary_

    Args:
        class_mapping (dict): _description_
        column_num (int): _description_

    Returns:
        _type_: _description_
    """
    name = list(class_mapping.keys())[list(class_mapping.values()).index(column_num)]
    mapping = {'others': 0, name: 1}
    return mapping

def get_hows_dataloaders(data_dir: str, batch_size: int, sequence: str):
    """_summary_

    Args:
        data_dir (str): _description_
        batch_size (int): _description_
        sequence (str): _description_

    Returns:
        _type_: _description_
    """
    # define data transforms 
    transform = transforms.Compose([
                transforms.Resize(280),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
    
    # define class mapping
    # class_mapping = {
    #     "apple": 0, "banana": 1, "bowl": 2, "egg": 3, "glass_bottle": 4, "milk": 5, "mug": 6, "pear": 7,
    #     "scissors": 8, "bread": 9, "can": 10, "fork": 11, "knife": 12, "pan": 13, "spoon": 14
    # }
    class_mapping = {"apple": 0, "banana": 1}

    # get the dataset
    mapping = multiclass_binaryclass(class_mapping, int(sequence))
    print("mapping", mapping)
    train_data = HOWS_CL_25(root=data_dir,
                            mode="train", 
                            batch_size=batch_size, 
                            sequence=int(sequence), 
                            num_workers=8,
                            transform=transform,
                            class_mapping=mapping)
    
    test_data = HOWS_CL_25(root=data_dir,
                           mode="validation", 
                           batch_size=batch_size, 
                           sequence=int(sequence), 
                           num_workers=8,
                           transform=transform,
                           class_mapping=mapping)
    dataloaders = (train_data, test_data, test_data)
    return dataloaders
