from __future__ import division
from __future__ import print_function

import warnings
import numpy as np
from sklearn.neighbors import NearestNeighbors
from sklearn.utils import check_array, check_random_state
from collections import Counter
__author__ = 'stavrianos'

# Link to paper:  bit.ly/22KgAnP

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
                 random_state=None,
                 verbose=True):
        """
        Initialize Adasyn object with required parameters
        """

        self.binary = binary
        self.ratio = ratio
        self.imb_threshold = imb_threshold
        self.k = k
        self.random_state = random_state
        self.verbose = verbose
        self.clstats = {}
        #self.warnings.filterwarning('error')
       # self.unique_classes_ = None

    def fit(self, X, y):
    	"""
    	Class method to define class populations and store them as instance
    	variables. Also stores majority class label
    	"""
    	self.X = check_array(X)
    	self.y = y
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


    def transform(self,X,y):
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
    	self.new_X, self.new_Y = self.oversample()

    	return self.new_X, self.new_Y

    def generate_samples(self,x,knns,knnLabels,cl):
        print('MY GI JOE')
        print(sum(self.gi))
        print(self.gi)
        # Matrix to store synthetically generated samples
        new_data = []
        for ind,elem in enumerate(x):
            # calculating k-neighbors that belong to minority (their indexes in x)
            print(knns[ind][1:-1],knnLabels[ind])
            # Unfortunately knn returns the example itself as a neighbor. So it needs
            # to be ignored thats why it is iterated [1:-1] and knnLabelsp[ind][+1]. 
            min_knns = [ele for index,ele in enumerate(knns[ind][1:-1]) if knnLabels[ind][index+1]==cl]
            if not min_knns:
                print('No k-neighbors belong to the minority class for row %s ,no samples produced' % elem)
                continue

            # generate gi synthetic examples for every minority example
            for i in range(0,int(self.gi[ind])):
                # randi holds an integer to choose from minority kNNs
                randi=self.random_state_.random_integers(0,len(min_knns)-1)
                l = self.random_state_.random_sample()
                #X[min_knns[randi]] is the Xzi on equation [5]
                si = self.X[elem] + (self.X[min_knns[randi]]-self.X[elem])*l
                new_data.append(si)
                

                print('Synthetic example: %s ' % si)
                print('Produced by initial: %s %s' % (self.X[elem], self.y[elem]))
                print('And brother:         %s %s' % (self.X[min_knns[randi]],self.y[min_knns[randi]]))
        print(np.asarray(new_data).shape)

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
                print('WE TALKING ABOUT CLASS %s' % cl)
                # G is the number of synthetic examples to be synthetically
                # produced for the current minority class
                self.G = (self.clstats[self.maj_class_] - self.clstats[cl]) * self.ratio

                # ADASYN is built upon eucliden distance so p=2 explicitly
                self.nearest_neighbors_ = NearestNeighbors(n_neighbors=self.k+1, p=2).fit(self.X)

                # keeping indexes of minority examples
                minx = [ind for ind,exam in enumerate(self.X) if y[ind]==cl]
                #print(minx, self.y[minx])
                # Computing kNearestNeighbors for every minority example
                knn = self.nearest_neighbors_.kneighbors(self.X[minx],return_distance=False)
               # print(y[knn])
                # Getting labels of k-neighbors of each example to determine how many of them 
                # are of different class than the one being oversampled
                knnLabels = [y[ele] for ind,ele in enumerate(knn)]


                tempdi = [Counter(i) for i in knnLabels]
               # print(tempdi)
                # Calculating ri as defined in ADASYN paper:
                # No. of k-neighbors belonging to different class than the minority divided by K 
                # which is ratio of friendly/non-friendly neighbors
                self.ri = np.array([(sum(i.values())-i[cl])/float(self.k) for i in tempdi])
                
                #Normalizing so that ri is a density distribution (i.e. sum(ri)=1)
                if np.sum(self.ri):
                    self.ri = self.ri / np.sum(self.ri)
                
                # Calculating #synthetic_examples that need to be generated for
                # each minority instance and rounding to nearest integer because
                # it can't produce e.g 2.35 new examples.
                self.gi = np.rint(self.ri * self.G)
              #  print(self.gi)
                #inter_X,inter_Y = self.generate_samples(minx,knn,knnLabels,cl)
                
                self.generate_samples(minx,knn,knnLabels,cl)
                #print(self.random_state_.rand(1))

        return(inter_X,inter_Y)










               



def deleteClass(X,y,num,c):
    """Delete 'num' samples from class=c in StudentLife dataset stress reports
    """
    
    twoIndex=np.array([i for i in range(len(y)) if y[i]==c])
    np.random.shuffle(twoIndex)


    delIndex=twoIndex[0:num]

    X=np.delete(X,delIndex,0)
    y=np.delete(y,delIndex,0)

    return(X,y)


from sklearn.datasets import load_iris
data = load_iris()
X = data['data']
y = data['target']
print(Counter(y))
#X,y = deleteClass(X,y,30,0)
X,y = deleteClass(X,y,28,2)
print(X.shape[1])
print(Counter(y))
#exit()
testnn = NearestNeighbors(n_neighbors=3, p=2).fit(X)
b =testnn.kneighbors(X,return_distance=False)
asda = ([y[ele] for ind,ele in enumerate(b) if y[ind]==1])

a=Adasyn(k=8,verbose=True)
a.fit_transform(X,y)