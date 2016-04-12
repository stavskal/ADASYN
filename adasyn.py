from __future__ import division
from __future__ import print_function
from __future__ import absolute_import
from __future__ import unicode_literals

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from collections import Counter

__author__ = 'stavrianos'

# Link to paper:  bit.ly/22KgAnP


class ADASYN(object):
    """
    Oversampling parent class with the main methods required by scikit-learn:
    fit, transform and fit_transform
    """

    def __init__(self,
                 ratio=0.5,
                 imb_threshold=0.5,
                 k=5,
                 random_state=None,
                 verbose=True):
        """
        :ratio:
            Growth percentage with respect to initial minority
            class size. For example if ratio=0.65 then after
            resampling minority class(es) will have 1.65 times
            its initial size


        :imb_threshold:
            The imbalance ratio threshold to allow/deny oversampling.
            For example if imb_threshold=0.5 then minority class needs
            to be at most half the size of the majority in order for
            resampling to apply

        :k:
            Number of K-nearest-neighbors

        :random_state:
            seed for random number generation

        :verbose:
            Determines if messages will be printed to terminal or not

        Extra Instance variables:

        :self.X:
            Feature matrix to be oversampled

        :self.y:
            Class labels for data

        :self.clstats:
            Class populations to determine minority/majority

        :self.unique_classes_:
            Number of unique classes

        :self.maj_class_:
            Label of majority class

        :self.random_state_:
            Seed
        """

        self.ratio = ratio
        self.imb_threshold = imb_threshold
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.clstats = {}
        self.num_new = 0
        self.index_new = []


    def fit(self, X, y):
        """
        Class method to define class populations and store them as instance
        variables. Also stores majority class label
        """
        self.X = check_array(X)
        self.y = np.array(y).astype(np.int64)
        self.random_state_ = check_random_state(self.random_state)
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

    def transform(self, X, y):
        """
        Applies oversampling transformation to data as proposed by
        the ADASYN algorithm. Returns oversampled X,y
        """
        self.new_X, self.new_y = self.oversample()

    def fit_transform(self, X, y):
        """
        Fits the data and then returns the transformed version
        """
        self.fit(X, y)
        self.new_X, self.new_y = self.oversample()

        self.new_X = np.concatenate((self.new_X, self.X), axis=0)
        self.new_y = np.concatenate((self.new_y, self.y), axis=0)

        return self.new_X, self.new_y

    def generate_samples(self, x, knns, knnLabels, cl):

        # List to store synthetically generated samples and their labels
        new_data = []
        new_labels = []
        for ind, elem in enumerate(x):
            # calculating k-neighbors that belong to minority (their indexes in x)
            # Unfortunately knn returns the example itself as a neighbor. So it needs
            # to be ignored thats why it is iterated [1:-1] and
            # knnLabelsp[ind][+1].
            min_knns = [ele for index,ele in enumerate(knns[ind][1:-1])
                         if knnLabels[ind][index +1] == cl]

            if not min_knns:
                continue

            # generate gi synthetic examples for every minority example
            for i in range(0, int(self.gi[ind])):
                # randi holds an integer to choose a random minority kNNs
                randi = self.random_state_.random_integers(
                    0, len(min_knns) - 1)
                # l is a random number in [0,1)
                l = self.random_state_.random_sample()
                # X[min_knns[randi]] is the Xzi on equation [5]
                si = self.X[elem] + \
                    (self.X[min_knns[randi]] - self.X[elem]) * l
                    
                new_data.append(si)
                new_labels.append(self.y[elem])
                self.num_new += 1

        return(np.asarray(new_data), np.asarray(new_labels))

    def oversample(self):
        """
        Preliminary calculations before generation of
        synthetic samples. Calculates and stores as instance
        variables: img_degree(d),G,ri,gi as defined by equations
        [1],[2],[3],[4] in the original paper
        """

        try:
            # Checking if variable exists, i.e. if fit() was called
            self.unique_classes_ = self.unique_classes_
        except:
            raise RuntimeError("You need to fit() before applying tranform(),"
                               "or simply fit_transform()")
        int_X = np.zeros([1, self.X.shape[1]])
        int_y = np.zeros([1])
        # Iterating through all minority classes to determine
        # if they should be oversampled and to what extent
        for cl in self.unique_classes_:
            # Calculate imbalance degree and compare to threshold
            imb_degree = float(self.clstats[cl]) / \
                self.clstats[self.maj_class_]
            if imb_degree > self.imb_threshold:
                if self.verbose:
                    print('Class %s is within imbalance threshold' % cl)
            else:
                # G is the number of synthetic examples to be synthetically
                # produced for the current minority class
                self.G = (self.clstats[self.maj_class_] - self.clstats[cl]) \
                            * self.ratio

                # ADASYN is built upon eucliden distance so p=2 default
                self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k + 1)
                self.nearest_neighbors_.fit(self.X)

                # keeping indexes of minority examples
                minx = [ind for ind, exam in enumerate(self.X) if self.y[ind] == cl]

                # Computing kNearestNeighbors for every minority example
                knn = self.nearest_neighbors_.kneighbors(
                    self.X[minx], return_distance=False)

                # Getting labels of k-neighbors of each example to determine how many of them
                # are of different class than the one being oversampled
                knnLabels = self.y[knn.ravel()].reshape(knn.shape)

                tempdi = [Counter(i) for i in knnLabels]

                # Calculating ri as defined in ADASYN paper:
                # No. of k-neighbors belonging to different class than the minority divided by K
                # which is ratio of friendly/non-friendly neighbors
                self.ri = np.array(
                    [(sum(i.values())- i[cl]) / float(self.k) for i in tempdi])

                # Normalizing so that ri is a density distribution (i.e.
                # sum(ri)=1)
                if np.sum(self.ri):
                    self.ri = self.ri / np.sum(self.ri)

                # Calculating #synthetic_examples that need to be generated for
                # each minority instance and rounding to nearest integer because
                # it can't produce e.g 2.35 new examples.
                self.gi = np.rint(self.ri * self.G)

                # Generation of synthetic samples
                inter_X, inter_y = self.generate_samples(
                                     minx, knn, knnLabels, cl)
                # in case no samples where generated at all concatenation
                # won't be attempted
                if len(inter_X):
                    int_X = np.concatenate((int_X, inter_X), axis=0)
                if len(inter_y):
                    int_y = np.concatenate((int_y, inter_y), axis=0)
        # New samples are concatenated in the beggining of the X,y arrays
        # index_new contains the indiced of artificial examples
        self.index_new = [i for i in range(0,self.num_new)]
        return(int_X[1:-1], int_y[1:-1])
