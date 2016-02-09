# Adaptive Synthetic Sampling Approach for Imbalanced Learning 

ADASYN is a python module that implements an adaptive oversampling technique for skewed datasets.

Many ML algorithms have trouble dealing with largely skewed datasets. If your dataset is 1000 examples and 950 of them belong to class 'Haystack' and the rest 50 belong to class 'Needle' it gets hard to predict new unseen data that belong to 'Needle' . What this algorithm does is create new artificial data that belong to the minority class by adding some semi-random noise to existing examples. For more information read the full paper

### Dependencies
* pip (needed for install)
* numpy
* scipy
* scikit-learn


### Installation

To use ADASYN you will need to clone this repository and install it running the following :

    git clone https://github.com/stavskal/ADASYN.git
    cd ADASYN
    pip install -r requirements.txt
    
    
After you have installed the packages you can proceed with using:

    from adasyn import ADASYN
    adsn = ADASYN(k=7,imb_threshold=0.6, ratio=0.75)
    new_X, new_y = adsn.fit_transform(X,y)  # your imbalanced dataset is in X,y

    
    
    
Original paper can be found [here](http://ieeexplore.ieee.org/xpl/login.jsp?tp=&arnumber=4633969&url=http://ieeexplore.ieee.org/xpls/abs_all.jsp%3Farnumber%3D4633969) 

This module implements the idea presented in the paper by Haibo He et al. and also includes oversampling for **multiclass** classification problems. It is designed to be compatible with [scikit-learn] (https://github.com/scikit-learn/scikit-learn). It focuses on oversampling the examples that are harder to classify and has shown results which sometimes outperform SMOTE or SMOTEboost.

An example can be seen below:

![alt tag](https://github.com/stavskal/ADASYN/blob/master/sample.jpg)


Props to [fmfn](https://github.com/fmfn) who implemented different oversampling techniques for his good code structure, which highly influenced this module, and documentation


Reference:

1. H. He, Y. Bai, E. A. Garcia, and S. Li, “ADASYN: Adaptive Synthetic Sampling Approach for Imbalanced Learning,” in Proc. Int. Joint Conf. Neural Networks (IJCNN’08), pp. 1322-1328, 2008.
