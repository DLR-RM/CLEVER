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

import cv2
import os
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import pickle

from scipy.ndimage import binary_dilation
from PIL import Image
from typing import Optional, Callable, Tuple, Any, Dict, List

from clever.tracker.aottracker import _palette
from clever.predictions import cartesian, barycentric


def crop_objs(pred_mask: np.ndarray, obj_id: int, frame: np.ndarray):
    """_summary_

    Args:
        pred_mask (np.ndarray): _description_
        obj_id (int): _description_
        frame (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    # get crop from current frame
    crop_mask = (pred_mask == obj_id).astype(np.uint8)
    x,y,w,h = cv2.boundingRect(crop_mask)
    crop_obj = frame * (pred_mask == obj_id).astype(np.uint8)[...,None]
    crop_obj = crop_obj[y:y+h,x:x+w]
    crop_obj = Image.fromarray(cv2.cvtColor(crop_obj, cv2.COLOR_BGR2RGB))
    return crop_obj

class WebcamImageProvider:
    def __init__(self):
        self.vid = cv2.VideoCapture(1) 
    
    def return_image(self) -> np.array:
        """_summary_

        Returns:
            np.array: we return an image with RGB format
        """
        ret, frame = self.vid.read()
        return cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    def node_closing(self):
        self.vid.release()

def extract_objects(pred_mask: np.ndarray,
                    frame: np.ndarray):
    """_summary_

    Args:
        pred_mask (np.ndarray): _description_
        frame (np.ndarray): _description_

    Returns:
        _type_: _description_
    """
    crop_objs = {}
    # Loop through objects detected in mask, unique id
    for id in np.unique(pred_mask):
        if id != 0:
            # Find object bounding box
            crop_mask = (pred_mask == id).astype(np.uint8)
            x,y,w,h = cv2.boundingRect(crop_mask)

            # Crop object from frame and save it
            crop = frame * (pred_mask == id).astype(np.uint8)[...,None]
            crop = crop[y:y+h,x:x+w]
            crop = Image.fromarray(cv2.cvtColor(crop, cv2.COLOR_BGR2RGB))
            crop_objs[id] = crop

    # Return list of cropped objects
    return crop_objs

def get_center_obj_id(pred_mask: np.ndarray,
                      center: Tuple[int, int],):
    '''
    Returns the id of the object that contains the center point.
    '''
    center_id = pred_mask[center[1], center[0]]
    return center_id

def draw_padding(img: np.ndarray, new_image_width: int, new_image_height: int):
    old_image_height, old_image_width, channels = img.shape

    # create new image of desired size and color (blue) for padding
    color = (0, 0, 0)
    result = np.full((new_image_height, new_image_width, channels), color, dtype=np.uint8)

    # compute center offset
    x_center = (new_image_width - old_image_width) // 2
    y_center = (new_image_height - old_image_height) // 2

    # copy img image into center of result image
    result[y_center:y_center+old_image_height, 
        x_center:x_center+old_image_width] = img
    return result

def draw_center_label(masked_frame: np.ndarray,
                      center: Tuple[int, int],
                      masked_labels_pred: Dict[int, str],
                      id: int):

    # Blue cross in centre of frame
    cv2.line(masked_frame, (center[0]-10,center[1]), (center[0]+10,center[1]), (255, 0, 0), 2)
    cv2.line(masked_frame, (center[0],center[1]-10), (center[0],center[1]+10), (255, 0, 0), 2)
    if id in masked_labels_pred:
        label = masked_labels_pred[id]
        if label == 'Unknown':
            # B G R
            color = (0,0,200) # red in BGR
        else: 
            color = (0,200,55) # green in BGR
    else:
        # blue in BGR
        color = (200,0,0) # blue in BGR
        label = 'Not segmented'

    # Put text with label of center object above the circle
    textsize = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.9, 2)[0]
    textX = int((20 - textsize[0]) / 2)
    textY = int((20 + textsize[1]) / 2)
    
    # rectangle around text bigger at least 5 pixels than text
    cv2.rectangle(masked_frame, (center[0]-10+textX-5,center[1]-10-textY-25), (center[0]-10+textX+textsize[0]+5,center[1]-10-textY+textsize[1]-10), (255,255,255), -1)
    cv2.putText(masked_frame, label, (center[0]-10+textX,center[1]-10-textY), cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    return masked_frame

def draw_masks_labels(masked_frame: np.ndarray, 
                      pred_mask: np.ndarray, 
                      masked_labels_pred: Dict[int, str],
                      alpha: float = 0.5):    
    for id in np.unique(pred_mask):
        crop_mask = (pred_mask == id).astype(np.uint8)
        if id in masked_labels_pred:
            if masked_labels_pred[id] == 'Unknown':
                # B G R
                color = (0,0,200) # red in BGR
            else: 
                color = (0,200,55) # green in BGR
        else:
            # blue in BGR
            color = (200,0,0) # blue in BGR

        # Fill with alpha=0.5 the object in the frame with the color
        masked_frame = cv2.addWeighted(masked_frame, 1, np.dstack((crop_mask*color[0], crop_mask*color[1], crop_mask*color[2])), alpha, 0)

        # Draw contour of the object in the frame
        contours, _ = cv2.findContours(crop_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(masked_frame, contours, -1, color, 2)
    return masked_frame

def save_prediction(pred_mask, output_dir, file_name):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask.save(os.path.join(output_dir, file_name))

def colorize_mask(pred_mask: np.ndarray):
    save_mask = Image.fromarray(pred_mask.astype(np.uint8))
    save_mask = save_mask.convert(mode='P')
    save_mask.putpalette(_palette)
    save_mask = save_mask.convert(mode='RGB')
    return np.array(save_mask)

def draw_mask(img: np.ndarray, mask: np.ndarray, alpha: float=0.5, id_countour: bool=False):
    img_mask = img
    if id_countour:
        # very slow ~ 1s per image
        obj_ids = np.unique(mask)
        obj_ids = obj_ids[obj_ids!=0]
        for id in obj_ids:
            # Overlay color on  binary mask
            if id <= 255:
                color = _palette[id*3:id*3+3]
            else:
                color = [0,0,0]
            foreground = img * (1-alpha) + np.ones_like(img) * alpha * np.array(color)
            binary_mask = (mask == id)
            # Compose image
            img_mask[binary_mask] = foreground[binary_mask]
            countours = binary_dilation(binary_mask,iterations=1) ^ binary_mask
            img_mask[countours, :] = 0
    else:
        binary_mask = (mask!=0)
        countours = binary_dilation(binary_mask, iterations=1) ^ binary_mask
        foreground = img*(1-alpha)+colorize_mask(mask)*alpha
        img_mask[binary_mask] = foreground[binary_mask]
        img_mask[countours,:] = 0
    return img_mask.astype(img.dtype)

def scatter(points: np.array, vertexlabels: Tuple[str, str, str] = None, **kwargs) -> matplotlib.figure.Figure:
    """Scatter plot of barycentric 2-simplex points on a 2D triangle.

    Args:
        points (np.array): (N, 3) shape array; N points on a 2-simplex
        vertexlabels : (str, str, str). Defaults to None.
                Labels for corners of the plot, in the order
                ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
                ``c == (0,0,1)``.
        **kwargs : any
            Arguments to :func:`plt.scatter`.

    Returns:
        matplotlib.figure.Figure: The drawn simplex plot figure
    """
    if vertexlabels is None:
        vertexlabels = ("1", "2", "3")

    projected = cartesian(points)
    plt.scatter(projected[:, 0], projected[:, 1], **kwargs)

    _draw_axes(vertexlabels)
    return plt.gcf()

def contour(f, vertexlabels: Tuple[str, str, str] = None, **kwargs) -> matplotlib.figure.Figure:
    """ Returns contour lines plot figure

    Args:
        f (function): Function to evaluate on (N, 3) ndarray of coordinates
        vertexlabels (str, str, str):         
            Labels for corners of the plot, in the order
            ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
            ``c == (0,0,1)``.. Defaults to None.
        **kwargs : any
            Arguments to :func:`plt.tricontour`.
    Returns:
        matplotlib.figure.Figure: The drawn contour line plot figure.
    """
    return _contour(f, vertexlabels, contourfunc=plt.tricontour, **kwargs)

def contourf(f, vertexlabels: Tuple[str, str, str] = None, **kwargs):
    """Filled contour plot on a 2D triangle of a function evaluated at
    barycentric 2-simplex points.

    Function signature is identical to :func:`contour` with the caveat that
    ``**kwargs`` are passed on to :func:`plt.tricontourf`."""
    return _contour(f, vertexlabels, contourfunc=plt.tricontourf, **kwargs)

def _contour(f, 
             vertexlabels: Tuple[str, str, str] = None, 
             contourfunc: matplotlib.figure.Figure = None, 
             **kwargs):
    """"Workhorse function for ``contour`` and ``contourf``.

    Args:
        f (function): Function to evaluate on (N, 3) ndarray of coordinates
        vertexlabels (str, str, str):         
            Labels for corners of the plot, in the order
            ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
            ``c == (0,0,1)``.. Defaults to None.
        contourfunc (function, optional): 
            The contour plotting function to use for actual plotting. Defaults to None.
    """
    if contourfunc is None:
        contourfunc = plt.tricontour
    if vertexlabels is None:
        vertexlabels = ("1", "2", "3")
    x = np.linspace(0, 1, 100)
    y = np.linspace(0, np.sqrt(3.0) / 2.0, 100)
    points2d = np.transpose([np.tile(x, len(y)), np.repeat(y, len(x))])
    points3d = barycentric(points2d)
    valid = (points3d.sum(axis=1) == 1.0) & ((0.0 <= points3d).all(axis=1))
    points2d = points2d[np.where(valid), :][0]
    points3d = points3d[np.where(valid), :][0]
    z = f(points3d)
    contourfunc(points2d[:, 0], points2d[:, 1], z, **kwargs)
    _draw_axes(vertexlabels)
    return plt.gcf()

def _draw_axes(vertexlabels: Tuple[str, str, str]):
    """Drawing of the axis

    Args:
        vertexlabels (str, str, str):         
            Labels for corners of the plot, in the order
            ``(a, b, c)`` where ``a == (1,0,0)``, ``b == (0,1,0)``,
            ``c == (0,0,1)``.
    """
    l1 = matplotlib.lines.Line2D([0, 0.5, 1.0, 0], [0, np.sqrt(3) / 2, 0, 0], color="k")
    axes = plt.gca()
    axes.add_line(l1)
    axes.xaxis.set_major_locator(matplotlib.ticker.NullLocator())
    axes.yaxis.set_major_locator(matplotlib.ticker.NullLocator())
    axes.text(-0.05, -0.05, vertexlabels[0])
    axes.text(1.05, -0.05, vertexlabels[1])
    axes.text(0.5, np.sqrt(3) / 2 + 0.05, vertexlabels[2])
    axes.set_xlim(-0.2, 1.2)
    axes.set_ylim(-0.2, 1.2)
    axes.set_aspect("equal")
    return axes