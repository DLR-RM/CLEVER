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
1. Take an instance of computing forward passes as tests.
2. Compare with no-bsbf and with bsbf.
3. Plot probabilities for one class.

Expectation: 
With BSBF we can take more smoothened predictions.
"""

import copy
import matplotlib.pyplot as plt
import numpy as np

# Get a basic implementation running on a toy setup.
# Test it for all the other clasees.
# TODO: test p2l and l2p.
# TODO: test the prior.
# TODO: test the filtering overall.
# TODO: test the reset functionality.
# TODO: integrate into the clever system.

class BSBF:
    """
    Implementation of binary state bayesian filter (BSBF)

    See Probabilistic Robotics of Burgard et al 2006.
    """
    def __init__(self, prior=0.5):
        self.prior = prior
        self.l_0 = self.a_priori()
        self.l_recursive = None

    def p2l(self, p_x):
        return np.log(p_x/(1-p_x))

    def l2p(self, l_x):
        return 1 - (1/(1+np.exp(l_x)))

    def filtering(self, p_x):
        if self.l_recursive is None:
            self.l_recursive = self.p2l(p_x)
            return self.l2p(self.l_recursive-self.l_0)
        else:
            l_inversemodel = self.p2l(p_x)
            l_x_1 = self.a_posteriori(self.l_recursive, l_inversemodel)
            self.l_recursive = copy.deepcopy(l_x_1)
            return self.l2p(l_x_1)

    def reset(self):
        self.l_recursive = None

    def a_priori(self):
        return self.p2l(self.prior)

    def a_posteriori(self, l_recursive, l_inversemodel):
        return l_recursive+l_inversemodel-self.l_0


def tests():
    # initiate the filter
    filters = BSBF()

    # given probability
    prob = 0.75
    odds = filters.p2l(prob)
    print("log-odds:", odds)
    print("probability:", filters.l2p(odds))


def main():
    # introduce noisy signals
    noises = np.random.normal(0.75, 0.000001, 10000)
    noises = [0.98, 0.98, 0.98, 0.98, 0.98, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1, 0.1]
    noises = [0.001, 1.0, 0.001, 1.0, 1.0, 1.0, 1.0, 0.001, 1.0, 0.001, 1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0]

    # initiate the filter
    filters = BSBF()

    # loop through the noise
    p_x = list()
    for noise in noises:
        p_x.append(filters.filtering(0.99*noise))

    plt.plot(noises)
    plt.plot(p_x)
    plt.ylim(0, 1.1)
    plt.show()
    
if __name__ == "__main__":
    main()