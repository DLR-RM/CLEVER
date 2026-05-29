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
Predictions for clever system.

1. https://github.com/ericsuh/dirichlet is adapted for dirichlet distribution.
Implmenetation of "Maximum likelihood estimation of dirichlet distribution
parameters" from CMU technical report.

2. I implemented "Inference over distribution of posterior class probabilities
for reliable Bayesian classificaiton and object-level perception" IEEE RA-L. 
"""

import sys
import scipy as sp
import scipy.stats as stats
import torch
import torch.nn as nn
import numpy as np

from PIL import Image
from typing import List, Dict, Union, Optional
from torch.utils.data import DataLoader
from numpy import (
    arange,
    array,
    asanyarray,
    asarray,
    diag,
    exp,
    isscalar,
    log,
    ndarray,
    ones,
    vstack,
    zeros,
)
from numpy.linalg import norm
from scipy.special import gammaln, polygamma, psi
from scipy.stats import dirichlet

MAXINT = sys.maxsize
euler = -1 * psi(1)  # Euler-Mascheroni constant

def predict_dataloader(model: nn.Module,
                       dataloader: DataLoader,
                       device: str,
                       sigma: Optional[int] = None):
    pred_labels = []
    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)
        pred_label, value = predict_maxlogit(inputs, model, sigma=sigma)
        pred_labels.append(pred_label)
    return pred_labels

def predict_masked_labels(model: nn.Module, 
                          img_dict: Dict[int, np.array],
                          masked_labels_pred: Dict[int, str],
                          data_transforms: Dict[str, torch.Tensor],
                          device: str,
                          sigma: int=2,):
    infer_batch = torch.stack([data_transforms['test'](img) for img in img_dict.values()])
    pred_labels = []
    for img in infer_batch:
        label, logit_max = predict_maxlogit(img.unsqueeze(0).to(device), model, sigma=sigma)
        pred_labels.append(label)
    for i, id in enumerate(img_dict.keys()):
        if id not in masked_labels_pred.keys():
            masked_labels_pred[id] = pred_labels[i]
        else:
            if masked_labels_pred[id] == 'Unknown':
                masked_labels_pred[id] = pred_labels[i]
    return masked_labels_pred

def predict_labels_point(model: nn.Module, 
                         img: Image,
                         data_transforms: Dict[str, torch.Tensor],
                         device: str,
                         sigma: int=2,):
    infer_img = data_transforms['test'](img)
    label, logit_max = predict_maxlogit(infer_img.unsqueeze(0).to(device), model, sigma=sigma)
    return label

def predict_maxlogit(inputs: torch.Tensor,
                     model: nn.Module,
                     sigma: int):
    sigma = sigma if sigma is not None else 2
    model.eval()
    with torch.no_grad():
        logits = model(inputs)
        preds = []
        for logit in logits:
            preds.append(logit.cpu().numpy()[0][0])
    pred_idx = np.argmin(preds,axis=0)
    pred_value = preds[pred_idx]
    mean_logit = model.networks_maxlogits[pred_idx][0]
    std_logit = model.networks_maxlogits[pred_idx][1]
    if pred_value <= mean_logit + sigma * std_logit:
        label = model.networks_labels[pred_idx][0]
    else:
        label = 'Unknown'
    return label, pred_value


class NotConvergingError(Exception):
    """Error when a successive approximation method doesn't converge
    """
    pass

def test(D1: np.array, D2: np.array, method: str ="meanprecision", maxiter: int = None):
    """Test for statistical difference between observed proportions.

    Args:
        D1 (np.array): (N1, K) shape array
        D2 (np.array): (N2, K) shape array
        NOTE:   Input observations. ``N1`` and ``N2`` are the number of observations,
                and ``K`` is the number of parameters for the Dirichlet distribution
                (i.e. the number of levels or categorical possibilities).
                Each cell is the proportion seen in that category for a particular
                observation. Rows of the matrices must add up to 1.
        method (str, optional):
            One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
            which to find MLE Dirichlet distribution. Default is
            ``'meanprecision'``, which is faster.. Defaults to "meanprecision".
        maxiter (int, optional):         
            Maximum number of iterations to take calculations. Default is
            ``sys.maxint``.. Defaults to None.

    Raises:
        ValueError: _description_

    Returns:
        D (float):
            Test statistic, which is ``-2 * log`` of likelihood ratios.
        p (float):
            p-value of test.
        a0 (np.array): (K,) shape array
        a1 (np.array): (K,) shape array
        a2 (np.array): (K,) shape array
            MLE parameters for the Dirichlet distributions fit to 
            ``D1`` and ``D2`` together, ``D1``, and ``D2``, respectively.
    """
    N1, K1 = D1.shape
    N2, K2 = D2.shape
    if K1 != K2:
        raise ValueError("D1 and D2 must have the same number of columns")

    D0 = vstack((D1, D2))
    a0 = mle(D0, method=method, maxiter=maxiter)
    a1 = mle(D1, method=method, maxiter=maxiter)
    a2 = mle(D2, method=method, maxiter=maxiter)

    D = 2 * (loglikelihood(D1, a1) + loglikelihood(D2, a2) - loglikelihood(D0, a0))
    return (D, stats.chi2.sf(D, K1), a0, a1, a2)

def pdf(alphas: np.array):
    """Returns a Dirichlet PDF function

    Args:
        alphas (np.array): (K,) shape array
            The parameters for the distribution of shape ``(K,)``.

    Returns:
        function: The PDF function, takes an ``(N, K)`` shape input and gives an
        ``(N,)`` output.
    """
    alphap = alphas - 1
    c = np.exp(gammaln(alphas.sum()) - gammaln(alphas).sum())

    def dirichlet(xs: np.array):
        """Dirichlet PDF

        Args:
            xs (np.array): The ``(N, K)`` shape input matrix

        Returns:
            np.array: (N,) shape array. Point value for PDF
        """
        return c * (xs ** alphap).prod(axis=1)

    return dirichlet

def meanprecision(a: np.array):
    """Mean and precision of a Dirichlet distribution.

    Args:
        a (np.array): (K,) shape array
            Parameters of a Dirichlet distribution.

    Returns:
        mean (np.array): (K,) shape array
            Means of the Dirichlet distribution. Values are in [0,1].
        precision (float):
            Precision or concentration parameter of the Dirichlet distribution.
    """
    s = a.sum()
    m = a / s
    return (m, s)

def loglikelihood(D: np.array, a: np.array):
    """Compute log likelihood of Dirichlet distribution, i.e. log p(D|a).

    Args:
        D (np.array): (N, K) shape array
                      ``N`` is the number of observations, ``K`` is the number of
                      parameters for the Dirichlet distribution.
        a (np.array): (K,) shape array
                      Parameters for the Dirichlet distribution.

    Returns:
        logl (float):
            The log likelihood of the Dirichlet distribution"
    """
    N, K = D.shape
    logp = log(D).mean(axis=0)
    return N * (gammaln(a.sum()) - gammaln(a).sum() + ((a - 1) * logp).sum())

def mle(D: np.array, tol: float = 1e-7, method: str = "meanprecision", maxiter: int = None):
    """ Iteratively computes maximum likelihood Dirichlet distribution
    for an observed data set, i.e. a for which log p(D|a) is maximum.

    Args:
        D (np.array): (N, K) shape array
            ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        tol (float, optional): If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged.. Defaults to 1e-7.
        method (str, optional): One of ``'fixedpoint'`` and ``'meanprecision'``, designates method by
            which to find MLE Dirichlet distribution. Default is
            ``'meanprecision'``, which is faster.. Defaults to "meanprecision".
        maxiter (int, optional): Maximum number of iterations to take calculations. Default is
            ``sys.maxint``. Defaults to None.

    Returns:
        a (np.array): (K,) shape array
            Maximum likelihood parameters for Dirichlet distribution.
    """
    if method == "meanprecision":
        return _meanprecision(D, tol=tol, maxiter=maxiter)
    else:
        return _fixedpoint(D, tol=tol, maxiter=maxiter)

def _fixedpoint(D: np.array, tol: float = 1e-7, maxiter: int = None):
    """Simple fixed point iteration method for MLE of Dirichlet distribution

    Args:
        D (np.array): (N, K) shape array
            ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        tol (float, optional): If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged.. Defaults to 1e-7.
        maxiter (int, optional): Maximum number of iterations to take calculations. Default is
            ``sys.maxint``. Defaults to None.

    Returns:
        a (np.array): (K,) shape array
            Fixed-point estimated parameters for Dirichlet distribution.
    """
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _ipsi(psi(a0.sum()) + logp)
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        "Failed to converge after {} iterations, values are {}.".format(maxiter, a1)
    )

def _meanprecision(D: np.array, tol: float = 1e-7, maxiter: int = None):
    """Mean/precision method for MLE of Dirichlet distribution

    Args:
        D (np.array): (N, K) shape array
            ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        tol (float_, optional): If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged. Defaults to 1e-7.
        maxiter (int, optional):  Maximum number of iterations to take calculations. Default is
            ``sys.maxint``. Defaults to None.

    Returns:
    a (np.array) : (K,) shape array
        Estimated parameters for Dirichlet distribution.
    """
    logp = log(D).mean(axis=0)
    a0 = _init_a(D)
    s0 = a0.sum()
    if s0 < 0:
        a0 = a0 / s0
        s0 = 1
    elif s0 == 0:
        a0 = ones(a0.shape) / len(a0)
        s0 = 1
    m0 = a0 / s0

    # Start updating
    if maxiter is None:
        maxiter = MAXINT
    for i in range(maxiter):
        a1 = _fit_s(D, a0, logp, tol=tol)
        s1 = sum(a1)
        a1 = _fit_m(D, a1, logp, tol=tol)
        m = a1 / s1
        # Much faster convergence than with the more obvious condition
        # `norm(a1-a0) < tol`
        if abs(loglikelihood(D, a1) - loglikelihood(D, a0)) < tol:
            return a1
        a0 = a1
    raise NotConvergingError(
        f"Failed to converge after {maxiter} iterations, " f"values are {a1}."
    )

def _fit_s(D: np.array, a0: np.array, logp: np.array, tol: float = 1e-7, maxiter: int = 1000):
    """Update parameters via MLE of precision with fixed mean.

    Args:
        D (np.array):  (N, K) shape array
            ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        a0 (np.array): (K,) shape array
            Current parameters for Dirichlet distribution
        logp (np.array): (K,) shape array
            Mean of log-transformed D across N observations
        tol (float, optional):  If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged.. Defaults to 1e-7.
        maxiter (int, optional): Maximum number of iterations to take calculations. Defaults to 1000.

    Returns:
        np.array: (K,) shape array
            Updated parameters for Dirichlet distribution.
    """
    s1 = a0.sum()
    m = a0 / s1
    mlogp = (m * logp).sum()
    for i in range(maxiter):
        s0 = s1
        g = psi(s1) - (m * psi(s1 * m)).sum() + mlogp
        h = _trigamma(s1) - ((m ** 2) * _trigamma(s1 * m)).sum()

        if g + s1 * h < 0:
            s1 = 1 / (1 / s0 + g / h / (s0 ** 2))
        if s1 <= 0:
            s1 = s0 * exp(-g / (s0 * h + g))  # Newton on log s
        if s1 <= 0:
            s1 = 1 / (1 / s0 + g / ((s0 ** 2) * h + 2 * s0 * g))  # Newton on 1/s
        if s1 <= 0:
            s1 = s0 - g / h  # Newton
        if s1 <= 0:
            raise NotConvergingError(f"Unable to update s from {s0}")

        a = s1 * m
        if abs(s1 - s0) < tol:
            return a

    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s1}")

def _fit_m(D: np.array, a0: np.array, logp: np.array, tol: float = 1e-7, maxiter: int = 1000):
    """Update parameters via MLE of mean with fixed precision s

    Args:
        D (np.array): ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.
        a0 (np.array): (K,) shape array, Current parameters for Dirichlet distribution
        logp (np.array): (K,) shape array; Mean of log-transformed D across N observations
        tol (float, optional): If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged.. Defaults to 1e-7.
        maxiter (int, optional): Maximum number of iterations to take calculations.. Defaults to 1000.

    Returns:
        np.array: (K,) shape array
            Updated parameters for Dirichlet distribution.
    """
    s = a0.sum()
    for i in range(maxiter):
        m = a0 / s
        a1 = _ipsi(logp + (m * (psi(a0) - logp)).sum())
        a1 = a1 / a1.sum() * s

        if norm(a1 - a0) < tol:
            return a1
        a0 = a1

    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"s is {s}")

def _init_a(D: np.array):
    """Initial guess for Dirichlet alpha parameters given data D

    Args:
        D (np.array): (N, K) shape array
            ``N`` is the number of observations, ``K`` is the number of
            parameters for the Dirichlet distribution.

    Returns:
        np.array: (K,) shape array
            Crude guess for parameters of Dirichlet distribution.
    """
    E = D.mean(axis=0)
    E2 = (D ** 2).mean(axis=0)
    return ((E[0] - E2[0]) / (E2[0] - E[0] ** 2)) * E

def _ipsi(y: np.array, tol: float = 1.48e-9, maxiter: int = 10):
    """Inverse of psi (digamma) using Newton's method. For the purposes
    of Dirichlet MLE, since the parameters a[i] must always
    satisfy a > 0, we define ipsi :: R -> (0,inf).

    Args:
        y (np.array): (K,) shape array; y-values of psi(x)
        tol (float, optional): If Euclidean distance between successive parameter arrays is less than
            ``tol``, calculation is taken to have converged.. Defaults to 1.48e-9.
        maxiter (int, optional): Maximum number of iterations to take calculations. Defaults to 10.

    Returns:
        np.array: (K,) shape array; Approximate x for psi(x).
    """
    y = asanyarray(y, dtype="float")
    x0 = np.piecewise(
        y,
        [y >= -2.22, y < -2.22],
        [(lambda x: exp(x) + 0.5), (lambda x: -1 /_fixedpoint (x + euler))],
    )
    for i in range(maxiter):
        x1 = x0 - (psi(x0) - y) / _trigamma(x0)
        if norm(x1 - x0) < tol:
            return x1
        x0 = x1
    raise NotConvergingError(f"Failed to converge after {maxiter} iterations, " f"value is {x1}")

def _trigamma(x):
    return polygamma(1, x)

def cartesian(points: np.array) -> np.array:
    """Converts array of barycentric coordinates on a 2-simplex to an array of
    Cartesian coordinates on a 2D triangle in the first quadrant.

    Args:
        points (np.array): (N, 3) shape array; Points on a 2-simplex.

    Returns:
        np.array: (N, 2) shape array
            Cartesian coordinate points.

    Examples
    --------
    >>> cartesian((1,0,0))
    array([0, 0])

    >>> cartesian((0,1,0))
    array([0, 1])

    >>> cartesian((0,0,1))
    array([0.5, 0.8660254037844386]) # == [0.5, sqrt(3)/2]
    """
    points = np.asanyarray(points)
    ndim = points.ndim  # will use this to have similar output shape to input
    if ndim == 1:
        points = points.reshape((1, points.size))
    d = points.sum(axis=1)  # in case values aren't normalized
    x = 0.5 * (2 * points[:, 1] + points[:, 2]) / d
    y = (np.sqrt(3.0) / 2) * points[:, 2] / d
    out = np.vstack([x, y]).T
    if ndim == 1:
        return out.reshape((2,))
    return out

def barycentric(points: np.array) -> np.array:
    """Inverse of :func:`cartesian`."""
    points = np.asanyarray(points)
    ndim = points.ndim
    if ndim == 1:
        points = points.reshape((1, points.size))
    c = (2 / np.sqrt(3.0)) * points[:, 1]
    b = (2 * points[:, 0] - c) / 2.0
    a = 1.0 - c - b
    out = np.vstack([a, b, c]).T
    if ndim == 1:
        return out.reshape((3,))
    return out


class binary2multiclass:
    """ Transfering Classifier Socres into Accurate Multiclass Probability Estimates
    by Bianca Zadrozny and Charles Elkan.

    For one-against-all scenario, calibrating each binary classifier and then
    normalizing afterwards was found to be sufficient.

    Least-squares and coupling methods from the paper can also be implemented.
    """
    def __init__(self, mode="normalization") -> None:
        self.mode = mode
        pass

    def normalization(self, per_class_prob: np.array):
        """ Given per class probabilities, we convert them
        to valid probabilities over all classes through normalization.

        Args:
            per_class_prob (np.array): Batch size x Class number array

        Returns:
            np.array: Batch size x Class number array
        """
        normalization = 1/np.sum(per_class_prob, axis=1)
        class_prob = np.multiply(per_class_prob, normalization[:, np.newaxis])
        return class_prob
    
    def clever_normalization(self, per_class_prob: np.array):
        """Clever normalization

        Problem 1: We need to normalize instead of taking the maximum.
        The problem is, what if both banana and apple are predicted with 100% confidence?
        Normalization provides means to then say, I am uncertain.

        Problem 2: If we blindly normalize, the problem lies when
        both banana and apple are predicted with 0% confidence. Then, standard
        normalization scheme will be ill-defined, or provide meaningless values.

        Problem 3: What we do is, only normalize amongst the instances
        that actually made predictions, e.g., apple or banana with 50% confidence.
        This will mitigate some of the issues, but might fail to capture if apple
        or banana were predicted with 45% confidence. It is moment to query but
        because of wrong mechanism, this will fail.

        Therefore, we take only classes with confidence above 25% and
        provide normalization within that set.

        Args:
            per_class_prob (np.array): _description_

        Returns:
            _type_: _description_
        """
        # check if the input class confidences is below 0.25
        # if so, put it to complete zero and then normalize.
        per_class_prob[per_class_prob < 0.25] = 0
        per_class_prob = self.normalization(per_class_prob)
        per_class_prob[np.isnan(per_class_prob)] = 0
        return per_class_prob

    def clever_simple(self, per_class_prob: np.array):
        """Simplest output.
        We take the maximum probabilities and output only that.
        If the probabilities are both equal 1 for certain heads
        we take them into 50% each so that retraining is possible.
        """
        raise NotImplementedError
        
    def least_squares(self):
        raise NotImplementedError
    
    def coupling(self):
        raise NotImplementedError
    

class DirichletTemporals:
    def __init__(self, 
                 num_heads: int = 2,
                 num_class: int = 2, 
                 num_samples: int = 100, 
                 mc_samples: int = 10, 
                 batch_size: int = 4,
                 device='cuda'):
        self.num_class = num_class
        self.num_heads = num_heads
        self.num_samples = num_samples
        self.mc_samples = mc_samples
        self.batch_size = batch_size
        self.device = device

        # initialize lambdas
        self.lambdas = 1/self.num_class * np.ones((self.num_heads, self.batch_size, self.num_class, self.num_samples))

        # initialize mc_sample and lambda combinations
        self.combos = [(x, y) for x in range(0, self.mc_samples) for y in range(0, self.num_samples)]

        # initialize dirichlet model per head?
        self.alpha_init = [[5, 5], [5, 5]]
        self.models = [[torch.distributions.dirichlet.Dirichlet(torch.tensor(self.alpha_init[i]).to(self.device))\
                         for j in range(0, self.num_class)] for i in range(0, self.num_heads)]
    
    def subsample(self):
        return np.asarray(self.combos)[np.random.randint(low=0, high=len(self.combos), size=(self.num_samples, ))]

    def predictions(self, per_class_probs, add_noise: bool = True):
        """_summary_

        Args:
            per_class_probs (List[np.array]): Head number of lists. Array with shape Batch x class (2) x samples

        Returns:
            NOTE: self.lambdas: np.array with Head x Batch x class (2) x samples
        """
        # process per_class_probs with different heads to get the likelihood
        clouds = list()
        for (x,y) in self.subsample():
            if add_noise:
                loglikelihood = torch.stack([torch.stack([\
                    self.models[i][0].log_prob(self.perturbation(per_class_probs[i][:, :, x])), \
                    self.models[i][1].log_prob(self.perturbation(per_class_probs[i][:, :, x]))], \
                    dim=1) for i in range(0, self.num_heads)]).cpu().detach().numpy()
            else:
                loglikelihood = torch.stack([torch.stack([self.models[i][0].log_prob(per_class_probs[i][:, :, x]), \
                                            self.models[i][1].log_prob(per_class_probs[i][:, :, x])], \
                                            dim=1) for i in range(0, self.num_heads)]).cpu().detach().numpy()

            # select the relevant lambdas TODO: re-examine this assumption 
            log_lamb_k = np.clip(np.log(self.lambdas[:, :, :, y]) + loglikelihood, -500, None)

            # remove log and normalize the confidences
            clouds.append(np.exp(log_lamb_k) * (1/np.sum(np.exp(log_lamb_k), axis=2)[:, :, np.newaxis]))
        
        self.lambdas = np.stack(clouds, axis=3)
        prob_lists_mean = [np.mean(self.lambdas, axis=3)[i,:,1] for i in range(0, self.num_heads)]
        prob_lists_var = [np.var(self.lambdas, axis=3)[i,:,1] for i in range(0, self.num_heads)]

        return prob_lists_mean, prob_lists_var
    
    def reset(self):
        self.lambdas = 1/self.num_class * np.ones((self.num_heads, self.batch_size, self.num_class, self.num_samples))
    
    def parameter_identification(self, column_num: int, input: np.array):
        # re-normalize iwth noise for for numerical stability
        confidences = self.normalizer(self.random_number_around(input))

        # compute mle
        alpha_target = mle(confidences)
        alpha_background = mle(1-confidences)

        # assigning dirichlet models
        if column_num + 1  > len(self.models):
            self.models.append([torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_target).to(self.device)), \
                    torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_background).to(self.device))])
        else:
            self.models[column_num][0] = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_target).to(self.device))
            self.models[column_num][1] = torch.distributions.dirichlet.Dirichlet(torch.tensor(alpha_background).to(self.device))

    def perturbation(self, input, scale=0.25):
        input = input + scale*torch.rand(input.shape).to(self.device)
        normalization = 1/torch.sum(input, dim=1).to(self.device)
        return torch.multiply(input, normalization[:, None])

    def random_number_around(self, input, scale=0.25):
        return input + scale*np.random.rand(input.shape[0], input.shape[1])
    
    def normalizer(self, input):
        normalization = 1/np.sum(input, axis=1)
        return np.multiply(input, normalization[:, np.newaxis])