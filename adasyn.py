from __future__ import division
from __future__ import print_function


import numpy as np

__author__ = 'stavrianos'


class Adasyn(object):
    """
    Oversampling parent class with the main methods required by scikit-learn:
    fit, transform and fit_transform
    """

    def __init__(self,
                 binary=False,
                 ratio=0.5,
                 imb_threshold=0.5,
                 verbose=True):
        """
        Initialize Adasyn object with required parameters. Works with default
        """

        self.binary = binary
        self.ratio = ratio
        self.imb_threshold = imb_threshold
        self.verbose = verbose

    def fit(self, X, y):

        unique_classes = set(self.y)

        # Checking if parameters are set correctly depending
        # on whether the dataset is binary / multiclass
        if len(unique_classes) == 1:
            raise RuntimeError("Only one class present, exiting...")
        elif len(unique_classes) == 2:
            if not self.binary:
                raise ValueError('Set binary=True for binary classification problems')
        else:
            if self.binary:
                raise ValueError('Set binary=False (default) for multiclass problems')
