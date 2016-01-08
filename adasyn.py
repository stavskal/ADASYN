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
        self.clstats = {}

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


        # Initialize all class populations with zero
        for element in unique_classes:
        	self.clstats[element] = 0

        # Count occurences of each class
        for element in self.y:
        	self.clstas[element] += 1

        # Find majority class 
        v = list(self.clstats.values())
        k = list(self.clstats.keys())
        maj_class = k[v.index(max(v))]

        # Remove majority class from set
        unique_classes.remove(maj_class)
        
      	for cl in unique_classes:
      		# Calculate imbalance degree and compare to threshold
      		imb_degree = float(self.clstats[cl]) / self.clstats[maj_class]
      	
      		if imb_degree > self.imb_threshold:
      			raise ValueError('Imbalance threshold not satisfied, try reducing imb_threshold parameter')
      		else:
      			self.new_X, self.new_Y = self.oversample()





    def oversample():
