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
Implementation of CLEVER:

A robotiC stream-based active Learner via classifiER using Bayesian Progressive Neural Networks

"Robotic Stream-based Active Learning for Robust Semantic Scene Understanding from Human Instructions"
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
from torchvision import transforms
from torch.utils.data import DataLoader

from bpnn.pnn import ProgressiveNeuralNetwork
from clever.trainer import pnn_fit_the_knowns, \
    pnn_fit_the_unknowns, bpnn_fit_the_unknowns, bpnn_fit_the_knowns
# from clever.tracker.segtracker import SegTracker
# from clever.tracker.model_args import aot_args,segtracker_args sam_args
from clever.model import ClassifierNet, clever_pnn, TheCleverNetwork
from clever.utils import extract_objects, \
    draw_mask, crop_objs, draw_padding
from clever.data.clever import CleverDataset
from clever.predictions import binary2multiclass, DirichletTemporals

# TODO: add doctrings and documentations.
# TODO: classification module + AL.


class TheSystem:
    """ The parent class for the networks.
    """
    def __init__(self, 
                 args: argparse.ArgumentParser,
                 device: str,
                 sam_gap: int = 1000,
                 num_seg: int = 4,
                 is_temporal: bool = False,
                 class_mapping: dict = {"apple": 0},
                 mc_samples: int = 30,
                 pairwise_coupling: str = "normalization"):
        # initialization
        self.args = args
        self.device = device
        self.sam_gap = sam_gap
        self.num_seg = num_seg
        self.mc_samples = mc_samples
        self.pairwise_coupling = pairwise_coupling
        self.is_temporal = is_temporal
        self.dirichlets = DirichletTemporals()
        self.prob_converter = binary2multiclass()
        self.metric={}
        self.bbox, self.image_ref, self.pred_mask_ref = None, None, None
        self.frame_idx = 0
        self.class_mapping = class_mapping
        self.sequence_mapping = copy.deepcopy(self.class_mapping)
        self.sequence_mapping = dict.fromkeys(self.sequence_mapping, 0)
        self.transform = transforms.Compose([
                         transforms.ToTensor(),
                         transforms.Resize((280, 280)),
                         transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
                         ])

        # load classifier and segmentation tracker
        self.classifier = self._load_classifier()
        self.segtracker = self._load_segmentor()

    def _load_classifier(self, 
                         last_layer_name: str = 'fc', 
                         lateral_connections: Optional[List[str]] = None):
        """Load an already trained classifier model from synthetic data.

        Args:
            last_layer_name (str, optional): _description_. Defaults to 'fc'.
            lateral_connections (Optional[List[str]], optional): _description_. Defaults to None.

        Returns:
            _type_: _description_
        """
        return None
    
    def _load_segmentor(self):
        """Load an already trained tracker model: segment and track anything with mobile sam backened.

        Returns:
            _type_: _description_
        """
        # segtracker = SegTracker(segtracker_args, sam_args, aot_args)
        segtracker = SAM(self.args.model_path + "./clever_v1/" + "sam2.1_t.pt")
        # segtracker.restart_tracker()
        return segtracker

    def assign_bbox(self, bbox: List):
        """For robustness of SAM or other instance segmentation
        one could use assign bbox. If SAM works well, then no need.

        bbox (List): [[x1, y1], [x2, y2]]
        """
        self.bbox = bbox

    def assign_references(self, image_acqusition):
        # let the user place the object inside the box
        # then press s while holding the camera
        while(True): 
            image = image_acqusition.return_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            # drawing the bounding box
            x1 = self.bbox[0][0]
            y1 = self.bbox[0][1]
            x2 = self.bbox[1][0]
            y2 = self.bbox[1][1]
            x = x1 / image.shape[0]
            y = y1 / image.shape[1]
            w = (x2 - x1) / image.shape[0]
            h = (y2 - y1) / image.shape[1]
            align_image = cv2.rectangle(
                image, (int(x1), int(y1)), (int(x2), int(y2)), (255,0,0), 5)

            align_image = cv2.circle(
                align_image, (int(align_image.shape[1]/2), int(align_image.shape[0]/2)), 
                radius=10, color=(0, 0, 0), thickness=-1
            )

            # display the image and quit
            cv2.imshow('Place the object inside the box. Object should lie on the dot. Then, press s and hold the camera', align_image)
            cv2.moveWindow('Place the object inside the box. Object should lie on the dot. Then, press s and hold the camera', 0, 0)
            if cv2.waitKey(1) & 0xFF == ord('s'): 
                break

        # let the user decide the best segmentation results
        # press s to select, and press q to go next  
        while(True):
            image = image_acqusition.return_image()
            image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

            #pred_mask, masked_frame = self.segtracker.seg_acc_bbox(image, self.bbox)
            masked_frame = self.segtracker(image, bboxes=boxes)

            masked_frame = cv2.circle(
                masked_frame, (int(masked_frame.shape[1]/2), int(masked_frame.shape[0]/2)), 
                radius=10, color=(0, 0, 0), thickness=-1
            )
            ## TODO: we need to change this to 

            # display the image and quit
            cv2.imshow('Press q to select next candidate. Press s to make a selection.', masked_frame)
            cv2.moveWindow('Press q to select next candidate. Press s to make a selection.', 0, 0)
            if cv2.waitKey(0) & 0xFF == ord('q'): 
                pass
            elif cv2.waitKey(0) & 0xFF == ord('s'): 
                break
            else:
                raise NotImplementedError

        self.image_ref = copy.deepcopy(image)              
        self.pred_mask_ref = copy.deepcopy(pred_mask)
    
    def _load_dataloader(self, object_name: str, random_selection: bool=False):
        """_summary_

        Args:
            object_name (str): _description_

        Returns:
            _type_: _description_
        """
        train_dataset = CleverDataset(root=self.args.pooldata_dir, 
                                      split='train', 
                                      category=object_name, 
                                      transform=self.transform, 
                                      random_selection=random_selection,
                                      sequence=str(self.sequence_mapping[object_name]))
        test_dataset = CleverDataset(root=self.args.pooldata_dir, 
                                     split='validation', 
                                     category=object_name, 
                                     transform=self.transform,
                                     random_selection=random_selection,
                                     sequence=str(self.sequence_mapping[object_name]))
        train_dataloader = DataLoader(dataset=train_dataset, batch_size=8, shuffle=True, num_workers=8)
        test_dataloader = DataLoader(dataset=test_dataset, batch_size=8, shuffle=False, num_workers=8)
        return (train_dataloader, test_dataloader, test_dataloader)

    def buffering(self, image_acqusition, buffer_size: int):
        """_summary_

        Args:
            image_acqusition (ImageProvder): _description_
            buffer_size (int): _description_
            open_visualization (bool, optional): _description_. Defaults to True.

        Returns:
            buffer (List[Tuple(class_string, class_probs)]): stored values for query decision.
        """
        class_string_buffer, class_probs_buffer = [], []
        
        for i in range(buffer_size):
            image = image_acqusition.return_image()
            pred_masks, cropped_images, class_string, class_probs = self.predictions(image.copy(), self.is_temporal)

            masked_frame = draw_mask(image, pred_masks)
            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            images = [masked_frame]

            class_string_dict, class_probs_dict = {}, {}

            try:
                for img, string, prob, obj_id in zip(cropped_images.values(), class_string, class_probs, cropped_images.keys()):
                    # if the shape is not the same as image, then
                    # pad them with black to fill all in.
                    img = np.array(img)
                    img = draw_padding(img, image.shape[1], image.shape[0])
                    if prob[1] < 0.75:
                        text = "unknown"
                    else:
                        text = string + str(obj_id) +  ':' + str(round(prob[1] * 100)) + '%'
                    cv2.putText(
                        img=img, text=text, 
                        org=(int(image.shape[0]/2), int(image.shape[1]/2+150)),
                        fontFace=cv2.FONT_HERSHEY_TRIPLEX,
                        fontScale=1, color=(0, 255, 0),
                        thickness=1)
                    
                    images.append(img)
                    class_string_dict[str(obj_id)] = string
                    class_probs_dict[str(obj_id)] = prob[1]

                class_string_buffer.append(class_string_dict)
                class_probs_buffer.append(class_probs_dict)

                window_name = 'Video segmentation'
                cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
                cv2.resizeWindow(window_name, 2*image.shape[1], image.shape[0])
                cv2.imshow(window_name, np.hstack([img for img in images])) 
                cv2.moveWindow(window_name, 0, 0)

                if cv2.waitKey(1) & 0xFF == ord('q'): 
                    break

                if i == int(buffer_size-1):
                    if self.is_temporal:
                        self.dirichlets.reset()
                    cv2.destroyAllWindows()
            except TypeError:
                continue
        
        # parsing the output buffer
        if not self.is_temporal:
            class_string, class_probs = {}, {}
            for obj_id in cropped_images.keys():
                string_per_id = []
                prob_per_id = []
                for string, prob in zip(class_string_buffer, class_probs_buffer):
                    if str(obj_id) in string.keys():
                        string_per_id.append(string[str(obj_id)])
                        prob_per_id.append(prob[str(obj_id)])
                if len(string_per_id) != 0:                    
                    # majority of string_per_id
                    string_id = self.most_frequent(string_per_id)
                    # indices for the majority
                    arr_index = np.where(np.asarray(string_per_id) == string_id)
                    # average the probabilities
                    string_prob = np.mean(np.asarray(prob_per_id)[arr_index])
                    class_string[str(obj_id)] = string_id
                    class_probs[str(obj_id)] = string_prob
        return (class_string, class_probs)
        
    def most_frequent(List):
        return max(set(List), key = List.count)

    def predictions(self, image: np.array, is_temporal: bool = False):
        """_summary_

        Args:
            image (np.array): _description_

        Returns:
            _type_: _description_
        """
        # obtain tracker mask and parse the results
        pred_masks = self._every_object_tracking(image, self.frame_idx)
        cropped_images = extract_objects(pred_masks, image)

        # obtain label for all objects
        if bool(cropped_images):
            class_string, class_probs = self._object_classification_on_images(cropped_images, is_temporal)
        else:
            class_string, class_probs = None, None

        # update tracker index
        self.frame_idx = self.frame_idx + 1
        return pred_masks, cropped_images, class_string, class_probs

    def _every_object_tracking(self, image: np.array, frame_idx: int):
        """_summary_

        Args:
            image (np.array): RGB images
            frame_idx (int): i-th frame number for deciding tracker modes

        Returns:
            _type_: _description_
        """
        # if points are not set
        if self.points is None:
            raise AttributeError
        result = self.segtracker(image, points=[self.points[0][0], self.points[0][1]], labels=[1])
        return result[0].cpu().numpy().masks.data.squeeze()
    
    
    # def _every_object_tracking(self, image: np.array, frame_idx: int):
    #     """_summary_

    #     Args:
    #         image (np.array): RGB images
    #         frame_idx (int): i-th frame number for deciding tracker modes

    #     Returns:
    #         _type_: _description_
    #     """
    #     if self.bbox is None:
    #         # Run segmentation every sam_gap frame
    #         if (frame_idx % self.sam_gap) == 0:
    #             pred_mask = self.segtracker.seg(image)

    #             # Empty cache
    #             torch.cuda.empty_cache()
    #             gc.collect()
                
    #             # If not first frame, track objects
    #             if frame_idx != 0:
    #                 # Track objects
    #                 track_mask = self.segtracker.track(image)

    #                 # Find new objects, and update tracker with new objects
    #                 new_obj_mask = self.segtracker.find_new_objs(track_mask, pred_mask)
    #                 pred_mask = track_mask + new_obj_mask

    #             # Add objects to tracker
    #             self.segtracker.add_reference(image, pred_mask)
    #         else:
    #             # Track objects
    #             pred_mask = self.segtracker.track(image, update_memory=False)
    #     else:
    #         if (frame_idx % self.sam_gap) == 0:
    #             # Add objects to tracker
    #             if self.image_ref is not None and self.pred_mask_ref is not None:
    #                 self.segtracker.add_reference(self.image_ref, self.pred_mask_ref)
    #             else:
    #                 raise NotImplementedError
            
    #         # Track objects
    #         pred_mask = self.segtracker.track(image, update_memory=False)

    #     return pred_mask
    
    def _object_classification_on_images(self, cropped_images: dict, is_temporal: bool = False):
        """_summary_

        Args:
            cropped_images (dict): _description_

        Returns:
            class_string: 1Dim Array of Batch B number with strings
            class_probs: 1Dim Array of Batch B number with probabilities
            entropy_array: Head H X Batch B array with entropy values 
        """ 
        raise NotImplementedError
    
    def active_learn(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            object_name (str): _description_
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader], optional): _description_. Defaults to None.
        """
        # update the classifier
        if object_name in list(self.class_mapping.keys()):
            self.learn_the_knowns(
                object_name=object_name, 
                column_num=self.class_mapping[object_name], 
                dataloader=dataloader,
                random_selection=random_selection,
                )
        else:
            self.learn_the_unknowns(
                object_name=object_name, 
                dataloader=dataloader,
                random_selection=random_selection,
                )

        # update the dirichlet temporal model
        if self.is_temporal:
            self.learn_dirichlet_temporals(
                object_name=object_name, 
                column_num=self.class_mapping[object_name], 
                dataloader=dataloader
                )
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        raise NotImplementedError
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            column_num (int): _description_
        """
        raise NotImplementedError

    def learn_dirichlet_temporals(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """_summary_
        TODO: when pbnn being tested, we should add additional module here to put into the correct probabilities.

        Args:
            object_name (str): _description_
            column_num (int): _description_
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader], optional): _description_. Defaults to None.
        """
        raise NotImplementedError
    
    def query_decisions(self, buffer, min_threshold: float = 0.5, max_threshold: float = 0.75):
        """ Query decisions based on probabilities

        Args:
            buffer (List): _description_.
            threshold (float): entropy threshold.
        """
        class_string, mean_probabilities = buffer

        if not self.is_temporal:
            is_query = (np.asarray(list(mean_probabilities.values())) > min_threshold) \
                & (np.asarray(list( mean_probabilities.values())) < max_threshold)
            discovered_objects = dict()
            discovered_objects['objects'] = list(class_string.values())
            discovered_objects['confidence'] = list(mean_probabilities.values())
            return is_query, discovered_objects
        else:
            max_probs = np.max(mean_probabilities, axis=1)
            is_query = (max_probs > min_threshold) & (max_probs < max_threshold)

            # catching a completely unknown object
            if (max_probs < min_threshold).all():
                is_query = [True]
            
            # packing the results
            discovered_objects = dict()
            discovered_objects['objects'] = list(class_string)
            discovered_objects['confidence'] = list(max_probs)
            return is_query, discovered_objects
    
    def _prompt_object_tracking(self):
        """ Function for recording the object data; 
        We use per center mode (initially show the results around)

        Raises:
            NotImplementedError: _description_
        """
        raise NotImplementedError
    
    def data_collection(self, image_acqusition, object_name, speech_engine=None):
        """_summary_

        Args:
            image_acqusition (ImageProvder): _description_
            object_name (str): _description_
        """
        # create a folder structure
        path_name = self.args.pooldata_dir + './' + object_name
        if not os.path.exists(path_name):
            os.makedirs(path_name)

        # if new object, add sesquence_mapping 
        if object_name in self.sequence_mapping:
            # check if that sequence exists
            masks_path = path_name + '/seq' +  str(self.sequence_mapping[object_name])
            if os.path.exists(masks_path):
                self.sequence_mapping[object_name] = self.sequence_mapping[object_name]+1
        else:
            self.sequence_mapping[object_name]=0
        
        # display the segmented data
        is_save = False
        is_target_obj = False
        target_objs = None
        frame_idx = 0

        # loop starts for data collection
        while True:
            image = image_acqusition.return_image()
            pred_masks = self._every_object_tracking(image.copy(), frame_idx)
            masked_frame = draw_mask(copy.deepcopy(image), pred_masks)
            masked_frame = cv2.cvtColor(masked_frame, cv2.COLOR_RGB2BGR)
            cv2.imshow('Press s to start saving.', masked_frame)
            cv2.moveWindow('Press s to start saving.', 0, 0)

            pressedKey = cv2.waitKey(1) & 0xFF
            if pressedKey == ord('q'):
                break
            elif pressedKey == ord('s'):
                is_save = True

            if speech_engine is not None:
                is_save = True
                output = speech_engine.data_saver_cue()
                print("Speech engine cue:", output)
                if output:
                    break

            if frame_idx > 80:
                print("CLEVER mind: The data collection finished. Collected 80 samples.")
                print("CLEVER mind: I pick 32 most informative and diverse samples.")
                break

            if is_save:
                if frame_idx==0:
                    print("CLEVER mind: The data collection started.")
                objs = np.unique(pred_masks)

                # draw an image of a circle
                for obj_id in objs:
                    crop = crop_objs(pred_mask=pred_masks, obj_id=obj_id, frame=cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
                    crop = np.array(crop)
                    crop = crop[:, :, ::-1].copy()
                    crop_mask = (pred_masks == obj_id).astype(np.uint8)

                    if crop_mask[int(image.shape[0]/2), int(image.shape[1]/2)] != 0 and is_target_obj is False:
                        target_objs = obj_id
                        is_target_obj = True
                    
                    if target_objs is not None and target_objs==obj_id:
                        masks_path = path_name + '/seq' +  str(self.sequence_mapping[object_name]) + '/' + object_name
                    else:
                        masks_path = path_name + '/seq' +  str(self.sequence_mapping[object_name]) + '/masks' + str(obj_id)
                    
                    if not os.path.exists(masks_path):
                        os.makedirs(masks_path)

                    cv2.imwrite(masks_path + '/' + object_name + '_' +  str(frame_idx) + '.jpg', crop)
                frame_idx = frame_idx +1
        cv2.destroyAllWindows()
    
    def reset(self):
        self.frame_idx = 0

    def most_frequent(self, List):
        return max(set(List), key = List.count)


class TheCleverSystem(TheSystem):
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
        pnn.load_full_state_dict(torch.load(self.args.model_pt_file))
        pnn.to(self.device)
        return pnn
    
    def _object_classification_on_images(self, cropped_images: dict, is_temporal: bool = False, column_num: int = 1):
        """_summary_

        Args:
            cropped_images (dict): _description_

        Returns:
            class_string: 1Dim Array of Batch B number with strings
            class_probs: 1Dim Array of Batch B number with probabilities
            entropy_array: Head H X Batch B array with entropy values 
        """ 
        infer_batch = torch.stack([self.transform(img) for img in cropped_images.values()]).to(self.device)
        padding_cut_index, padding_cut_list = None, []
        if infer_batch.shape[0] != self.num_seg:
            if infer_batch.shape[0] > self.num_seg:
                self.num_seg = infer_batch.shape[0]
                # raise NotImplementedError
            if infer_batch.shape[0] < self.num_seg:
                for i in range(0, int(self.num_seg-infer_batch.shape[0])):
                    infer_batch = torch.cat([infer_batch, infer_batch[0, :, :, :].unsqueeze(0)])
                    padding_cut_list.append(i + 1)

        if len(padding_cut_list) != 0:
            padding_cut_index = max(padding_cut_list) 
        
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier(infer_batch)
            probs = [F.softmax(logit, dim=-1) for logit in logits]

            if self.pairwise_coupling == "normalization":
                if is_temporal: 
                    # compute temporal probabilities
                    # check the shape of probs / MC-integration
                    try:
                        probs[0].shape[2] == self.mc_samples
                    except:
                        probs = [torch.stack([prob for i in range(0, self.mc_samples)], dim=2) for prob in probs]
                    
                    # compute the predictions using dirichlet model
                    prob_lists_mean, _ = self.dirichlets.predictions(probs)
                else:
                    prob_lists_mean = [prob[:, 1].cpu().detach().numpy() for prob in probs]

                # normalizing probabilities
                if column_num is not None:
                    prob_array = [prob.cpu().detach().numpy() for prob in probs][column_num]
                    class_string = np.array(
                        [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(column_num)]
                        for i in range(self.num_seg)]
                        )
                else:
                    prob_array = self.prob_converter.clever_normalization(np.stack(prob_lists_mean).T) 

                    # argmax to find the classes
                    class_numbers = np.argmax(prob_array, axis=1)
                    class_string = np.array(
                        [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(class_number)]
                        for class_number in class_numbers]
                        )
            else:
                raise AttributeError
        
        if padding_cut_index is not None:
            class_string = class_string[:-padding_cut_index]
            prob_array = prob_array[:-padding_cut_index]
            
        return class_string, prob_array
    
    def learn_the_unknowns(self, object_name: str, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader]): _description_
            object_name (str): _description_
        """
        # update the class mapping
        self.class_mapping.update({object_name: int(list(self.class_mapping.values())[-1]+1)})
        self.sequence_mapping[object_name] = 0
        column_num = len(self.class_mapping)

        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name, random_selection)

        # update the classifier
        self.metric, self.classifier = pnn_fit_the_unknowns(model=self.classifier,
                                                            dataloader=dataloader,
                                                            device=self.device,
                                                            use_validation_set=False,
                                                            num_epochs=1,
                                                            is_classification=[True for i in range(column_num)])
        print("CLEVER mind: Learning the unknowns with metric ", self.metric)
    
    def learn_the_knowns(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None, random_selection: bool = False):
        """_summary_

        Args:
            dataloader (Tuple[DataLoader, Otional[DataLoader], DataLoader]): _description_
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
        print("CLEVER mind: Learning the knowns with metric ", self.metric)

    def learn_dirichlet_temporals(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """_summary_

        Args:
            object_name (str): _description_
            column_num (int): _description_
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader], optional): _description_. Defaults to None.
        """
        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name)

        # create the dataloader
        (train_dataloader, _, _) = self._load_dataloader(object_name)

        # compute the probabilities of only for correct examples
        self.classifier.eval()
        confidences = list()
        with torch.no_grad():
            for inputs, targets in train_dataloader:
                # predictions 
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.classifier(inputs)
                probs = [F.softmax(logit, dim=-1) for logit in logits][column_num]
                class_numbers = torch.argmax(probs, dim=1)

                # check if correct examples are provided
                condition = targets == class_numbers
                indices = condition.nonzero()

                # store the variables
                if len(probs[indices, :]) != 0 and len(indices) != 0 and len(probs[indices, :].squeeze().shape) == 2:                        
                    # store the variables
                    confidences.append(probs[indices, :].squeeze().cpu().detach().numpy())

        # compute the mle estimates
        self.dirichlets.parameter_identification(column_num=column_num, input=np.concatenate(confidences))


class TheMostCleverSystem(TheSystem):
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

        # declare the model
        bpnn = TheCleverNetwork(model=pnn, priors=None, backbone=pnn.backbone, weight_decay=1e-5)
        if self.args.prior_pt_file is not None:
            bpnn.load_full_state_dict(torch.load(self.args.prior_pt_file))
        bpnn.to(self.device)
        return bpnn
    
    def _object_classification_on_images(self, cropped_images: dict, is_temporal: bool = False, column_num: int = None):
        """_summary_

        Args:
            cropped_images (dict): _description_

        Returns:
            class_string: 1Dim Array of Batch B number with strings
            class_probs: 1Dim Array of Batch B number with probabilities
            entropy_array: Head H X Batch B array with entropy values 
        """ 
        # pad to ensure the same amount of batches for inference # TODO: there is something wrong here!
        infer_batch = torch.stack([self.transform(img) for img in cropped_images.values()]).to(self.device)
        padding_cut_index, padding_cut_list = None, []
        if infer_batch.shape[0] != self.num_seg:
            if infer_batch.shape[0] > self.num_seg:
                self.num_seg = infer_batch.shape[0] 
            if infer_batch.shape[0] < self.num_seg:
                for i in range(0, int(self.num_seg-infer_batch.shape[0])):
                    infer_batch = torch.cat([infer_batch, infer_batch[0, :, :, :].unsqueeze(0)])
                    padding_cut_list.append(i + 1)
        if len(padding_cut_list) != 0:
            padding_cut_index = max(padding_cut_list) 
        
        # predictions
        self.classifier.eval()
        with torch.no_grad():
            logits = self.classifier.bayesian_analysis(infer_batch, num_samples=self.mc_samples)
            probs = [F.softmax(logit.permute(0, 2, 1), dim=-1) for logit in logits] 
            
            if is_temporal: 
                # compute temporal probabilities
                # check the shape of probs / MC-integration
                try:
                    probs[0].shape[1] == self.mc_samples
                except:
                    probs = [torch.stack([prob for i in range(0, 10)], dim=1) for prob in probs]
                
                # compute the predictions using dirichlet model
                prob_lists_mean, _ = self.dirichlets.predictions(probs)
            else:
                prob_lists_mean = [np.mean(prob[:, :, 1].cpu().detach().numpy(), axis=1) for prob in probs]

            # normalizing probabilities
            if column_num is not None:
                prob_array = [np.mean(prob.cpu().detach().numpy(), axis=1) for prob in probs][column_num]
                class_string = np.array(
                    [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(column_num)]
                    for i in range(self.num_seg)]
                    )
            else:
                prob_array = np.stack(prob_lists_mean).T

                # argmax to find the classes
                class_numbers = np.argmax(prob_array, axis=1) # TODO: here it doesnt make sense. Also, it doesnt align with normal CLEVER

                # probabilities and the classes
                prob_array = [np.mean(prob.cpu().detach().numpy(), axis=1) for prob in probs]
                prob_list = []
                for idx, num in enumerate(class_numbers):
                    prob_list.append(prob_array[num][idx, :])
                prob_array = np.stack(prob_list)
                class_string = np.array(
                    [list(self.class_mapping.keys())[list(self.class_mapping.values()).index(class_number)]
                    for class_number in class_numbers]
                    )

        if padding_cut_index is not None:
            class_string = class_string[:-padding_cut_index]
            prob_array = prob_array[:-padding_cut_index]
            
        return class_string, prob_array
    
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
        print("CLEVER mind: Learning the unknowns with metric ", self.metric)
    
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
        self.metric, self.classifier = bpnn_fit_the_knowns(model=self.classifier, 
                                                           dataloader=dataloader,
                                                           train_dataset=dataloader[0],
                                                           column_num=column_num)
        print("CLEVER mind: Learning the knowns with metric ", self.metric)

    def learn_dirichlet_temporals(self, object_name: str, column_num: int, dataloader: Tuple[DataLoader, Optional[DataLoader], DataLoader] = None):
        """_summary_

        Args:
            object_name (str): _description_
            column_num (int): _description_
            dataloader (Tuple[DataLoader, Optional[DataLoader], DataLoader], optional): _description_. Defaults to None.
        """
        # create the dataloader
        if dataloader is None:
            dataloader = self._load_dataloader(object_name)

        # create the dataloader
        (train_dataloader, _, _) = self._load_dataloader(object_name)

        # compute the probabilities of only for correct examples
        self.classifier.eval()
        confidences = list()
        with torch.no_grad():
            for inputs, targets in train_dataloader:
                # predictions 
                # TODO: could be that you have to integrate the mean, not the distribution which is obsecure.
                inputs, targets = inputs.to(self.device), targets.to(self.device)
                logits = self.classifier.bayesian_analysis(inputs, num_samples=self.mc_samples)
                probs = [F.softmax(logit, dim=-1) for logit in logits][column_num]

                for i in range(self.mc_samples):
                    prob = probs[:,:,i]
                    class_numbers = torch.argmax(prob, dim=1)

                    # check if correct examples are provided
                    condition = targets == class_numbers
                    indices = condition.nonzero()

                    if len(prob[indices, :]) != 0 and len(indices) != 0 and len(prob[indices, :].squeeze().shape) == 2:                        
                        # store the variables
                        confidences.append(prob[indices, :].squeeze().cpu().detach().numpy())

        # compute the mle estimates
        self.dirichlets.parameter_identification(column_num=column_num, input=np.concatenate(confidences))

    def save(self):
        torch.save(self.classifier.full_state_dict(), os.path.join(self.args.model_path, './clever_v1/full_state_dict_bpnn.pt'))

    def load(self):
        self.classifier = self._load_classifier()
