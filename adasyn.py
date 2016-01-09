from __future__ import division
from __future__ import print_function


import numpy as np
from sklearn.neighbors import NearestNeighbors
from collections import Counter
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
                 k=5,
                 verbose=True):
        """
        Initialize Adasyn object with required parameters
        """

        self.binary = binary
        self.ratio = ratio
        self.imb_threshold = imb_threshold
        self.k = k
        self.verbose = verbose
        self.clstats = {}
       # self.unique_classes_ = None

    def fit(self, X, y):
    	"""
    	Class method to define class populations and store them as instance
    	variables. Also stores majority class label
    	"""
    	self.X = X
    	self.y = y

        self.unique_classes_ = set(self.y)
        
        # Initialize all class populations with zero
        for element in self.unique_classes_:
            self.clstats[element] = 0

        # Count occurences of each class
        for element in self.y:
            self.clstats[element] += 1

        # Find majority class
        v = list(self.clstats.values())
        k = list(self.clstats.keys())
        self.maj_class_ = k[v.index(max(v))]

        if self.verbose:
            print(
                'Majority class is %s and total number of classes is %s' 
                % (self.maj_class_, len(self.unique_classes_)))


    def transform(self):
    	"""
    	Applies oversampling transformation to data as proposed by
    	the ADASYN algorithm. Returns oversampled X,y
    	"""
    	self.new_X , self.new_Y = self.oversample()


    def fit_transform(self,X,y):
    	"""
    	Fits the data and then returns the transformed version
    	"""
    	self.fit(X,y)
    	self.new_X, self.new_Y = self.transform()

    	return self.new_X, self.new_Y



    def oversample(self):
        """
        Actual generation of synthetic data, called by transform
        and fit_transform
        """

    	try:
    		# Removing majority class from set since it is
        	# stored in its own variable
        	self.unique_classes_ = self.unique_classes_ 
        except:
    		raise RuntimeError("You need to fit() before applying tranform(),"
    							"or simply fit_transform()")

    	# Checking if parameters are set correctly depending
        # on whether the labels are binary / multiclass
    	if len(self.unique_classes_) == 1:
            raise RuntimeError("Only one class present, exiting...")
        elif len(self.unique_classes_) == 2:
            if not self.binary:
                raise ValueError('Set binary=True for binary classification')
        else:
            if self.binary:
                raise ValueError('Set binary=False(default)for multiple '
                                 'classes')
        
        
        # Iterating through all minority classes to determine
        # if they should be oversampled and to what extent
    	for cl in self.unique_classes_:
            # Calculate imbalance degree and compare to threshold
            imb_degree = float(self.clstats[cl]) / self.clstats[self.maj_class_]
            if imb_degree > self.imb_threshold:
            	if self.verbose:
            		print('Class %s is within imbalance threshold' % cl)
            else:
                # G is the number of synthetic examples to be synthetically
                # produced for the current minority class
                G = (self.clstats[self.maj_class_] - self.clstats[cl]) * self.ratio
                
                # ADASYN is built upon eucliden distance so p=2 explicitly
                self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k, p=2).fit(self.X)

                # keeping indexes of minority examples
                minx = [ind for ind,exam in enumerate(self.X) if y[ind]==cl]
               
                # Computing kNearestNeighbors for every minority example
                knn = self.nearest_neighbors_.kneighbors(self.X[minx],return_distance=False)
                
                # Getting labels of k-neighbors of each example to determine how many of them 
                # are of different class than the one being oversampled
                knnLabels = [y[ele] for ind,ele in enumerate(knn) if y[ind]!=cl]
                tempdi = [Counter(i) for i in knnLabels]

                # Calculating ri as defined in ADASYN paper:
                # No. of neighbors belonging to different class than the minority divided by K 
                # which is ratio of friendly/non-friendly neighbors 
                self.ri = [(sum(i.values())-i[cl])/float(self.k) for i in tempdi]
                print(self.ri) #not normalized yet



               






from sklearn.datasets import load_iris
data = load_iris()
X = data['data']
y = data['target']
testnn = NearestNeighbors(n_neighbors=3, p=2).fit(X)
b =testnn.kneighbors(X,return_distance=False)
asda = ([y[ele] for ind,ele in enumerate(b) if y[ind]==1])
#print(asda)
#print(b[1])
#exit()
a=Adasyn(imb_threshold=1,verbose=True)
a.fit_transform(X,y)